"""
Copyright 2024 Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import traceback
from copy import copy, deepcopy
from sqlite3 import InternalError

import dask
from dask.base import tokenize
from dask.core import flatten, get_dependencies, reverse_dict
from dask.delayed import Delayed
from dask.optimization import cull, fuse, inline
from dask.utils import (
    apply,
)

from .core import KEY_SEP, Node


def base_name(name):
    return name.split(KEY_SEP)[0]


FUNCTION = 1
DATA = 2
REQUEST = 3

# FUNCTION = 0
# DATA = 1
# REQUEST = 2


class App:
    __conf = {
        "fail_mode": "fail",
        "use_delayed": False,
    }
    __setters = ["fail_mode", "use_delayed"]

    @staticmethod
    def config(name):
        return App.__conf[name]

    @staticmethod
    def exists(name):
        return name in App.__conf

    @staticmethod
    def set(name, value):
        if name in App.__setters:
            App.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")


def setting_exists(key):
    return App.exists(key)


def get_setting(key):
    return App.config(key)


def set_setting(key, value):
    return App.set(key, value)


def dict_update(base, update, convert_nestedfrozen=False):
    if not isinstance(base, dict) or not isinstance(update, dict):
        raise TypeError(
            f"dict_update requires two dicts as input. But we received {type(base)} and {type(update)}"
        )

    for key in update:
        if isinstance(base.get(key), dict) and isinstance(update[key], dict):
            if convert_nestedfrozen:
                base[key] = dict(base[key])
            base[key] = dict_update(
                base[key], update[key], convert_nestedfrozen=convert_nestedfrozen
            )
        else:
            base[key] = update[key]

    return base


class NodeFailedException(Exception):
    def __init__(self, exception=None):
        """The default exception when a node's compute function fails and failsafe mode
        is enable, i.e. the global setting `fail_mode` is not set to `fail`.
        this exception is caught by the deklare processing system and depending on the
        global variable `fail_mode`, leads to process interruption or continuation.

        Args:
            exception (any, optional): The reason why it failed, e.g. another exception.
                Defaults to None.
        """
        if get_setting("fail_mode") == "warning" or get_setting("fail_mode") == "warn":
            print(exception)
        self.exception = exception

    def __str__(self):
        return str(self.exception)


class FailSafeWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except Exception:
            trace = traceback.format_exc(2)
            return NodeFailedException(trace)


def update_key_in_config(request, old_key, new_key):
    if "config" in request:
        if "keys" in request["config"]:
            if old_key in request["config"]["keys"]:
                request["config"]["keys"][new_key] = request["config"]["keys"].pop(
                    old_key
                )


def generate_clone_key(current_node_name, to_clone_key, clone_id):
    return (
        base_name(to_clone_key)
        + KEY_SEP
        + tokenize([current_node_name, to_clone_key, "deklare_clone", clone_id])
    )


# @profile
def configuration(
    delayed,
    request,
    keys=None,
    default_merge=None,
    optimize_graph=True,
    dependants=None,
    clone_instead_merge=True,
):
    """Configures each node of the graph by propagating the request from outputs
    to inputs. Each node checks if it can fulfil the request and what it needs to fulfil
    the request. If a node requires additional configurations to fulfil the request it
    can set the 'requires_request' flag in the returned request and this function will
    add the return request as a a new input to the node's __call__().
    See also Node.configure()

    Args:
        delayed (dask.delayed or list): Delayed object or list of delayed objects
        request (dict or list): request (dict), list of requests
        keys (_type_, optional): _description_. Defaults to None.
        default_merge (_type_, optional): _description_. Defaults to None.
        optimize_graph (bool, optional): _description_. Defaults to True.
        dependants (_type_, optional): _description_. Defaults to None.

    Raises:
        RuntimeError: If graph cannot be configured

    Returns:
        dask.delayed: The configured graph
    """

    if not isinstance(delayed, list):
        collections = [delayed]
    else:
        collections = delayed

    # dsk = dask.base.collections_to_dsk(collections)
    dsk, dsk_keys = dask.base._extract_graph_and_keys(collections)
    if dependants is None:
        _, dependants = dask.core.get_deps(dsk)

    # dsk_dict = {k:dsk[k] for k in dsk.get_all_external_keys()}
    dsk_dict = {k: dsk[k] for k in dsk.keys()}

    if keys is None:
        keys = dsk_keys
    if not isinstance(keys, (list, set)):
        keys = [keys]

    work = list(set(flatten(keys)))
    # create a deepcopy, otherwise we might overwrite requests and falsify its usage outside of this function
    # request = deepcopy(request)
    if isinstance(request, list):
        # request = [NestedFrozenDict(r) for r in request if r]
        request = [r for r in request if r]
        if len(request) != len(work):
            raise RuntimeError(
                "When passing multiple request items "
                "The number of request items must be same "
                "as the number of keys"
            )

        # For each output node different request has been provided
        requests = {work[i]: [request[i]] for i in range(len(request))}
    else:
        # request = NestedFrozenDict(request)
        # Every output node receives the same request
        requests = {k: [request] for k in work}

    remove = {k: False for k in work}
    input_requests = {}
    # We will create a new graph with the configured nodes of the old graph
    # out_keys keeps track of the keys we have configured and
    # remember them for assembling the new graph
    out_keys = []

    # using dict here for performance because we are doing `if key in work` later
    # (and not using sets because for unknown reasons it doesn't work)
    work = {k: True for k in work}

    def normalize_node(k):
        # any node that does not have the following structure will get it
        # (apply, func, args, kwargs)
        if dsk_dict[k][0] is not apply:
            dsk_dict[k] = (apply, dsk_dict[k][0], list(dsk_dict[k][1:]), {})

    while work:
        # new_work = []
        new_work = dict()

        out_keys += work
        for k in work:
            # if k not in requests:
            #     # there wasn't any request stored use initial config
            #     requests[k] = [config]

            # check if we have collected all dependencies so far
            # we will come back to this node another time
            # TODO: make a better check for the case when dependants[k] is a set, also: why is it a set in the first place..?
            if (
                k in dependants
                # and len(dependants[k]) != len(requests[k])
                and not isinstance(dependants[k], set)
            ):
                continue

            if k not in requests:
                InternalError(f"Failed to find request for node {k}")

            # set configuration for this node k
            argument_is_node = None
            if isinstance(dsk_dict[k], tuple):
                # any node that does not have the following structure will get it
                # (apply, func, args, kwargs)
                normalize_node(k)

                # now every node will have the function in the second item of the tuple
                if hasattr(dsk_dict[k][FUNCTION], "__self__"):
                    if isinstance(dsk_dict[k][FUNCTION].__self__, Node):
                        argument_is_node = 1
            # Check if we get a node of type Node class
            if argument_is_node is not None:
                # Yes, we've got a node class so we can use it's configure function
                assert len(requests[k]) == 1
                current_request = requests[k][0]
                new_request = dsk_dict[k][argument_is_node].__self__.configure(
                    current_request
                )  # Call the class configuration function
                # if not isinstance(new_request, list):
                #     new_request = [new_request]
                # # convert back to dicts (here we are allowed to modify it)
                # new_request = [dict(r) for r in new_request]
            else:
                # We didn't get a Node class so there is no
                # custom configuration function: pass through
                new_request = {}
                assert len(requests[k]) == 1
                r = dict(requests[k][0])
                if r:
                    # sanitize request
                    r.pop("requires_request", None)
                    r.pop("insert_predecessor", None)
                    r.pop("clone_dependencies", None)
                    r.pop("remove_dependency", None)
                    r.pop("remove_dependenies", None)
                    new_request = r

            # update dependencies
            # we're going to get all dependencies of this node
            # then, we'll check if this node requires to clone it's input path
            # If so, each cloned path gets a different request from this node
            # (these are contained in a list in `clone_dependencies`)
            # we are going to introduce new keys and new nodes in the graph
            # therefore we must update this nodes input keys (hacking it from/to dsk_dict[k][DATA])
            # for each clone

            # For now it's not possible to have predecessors and multiple requests
            # User must use one request with `clone_dependencies` and `insert_predecessor` keys
            insert_predecessor = []
            if (
                new_request is not None
                and not isinstance(new_request, list)
                and "insert_predecessor" in new_request
                and new_request["insert_predecessor"]
            ):
                insert_predecessor = new_request["insert_predecessor"]
                del new_request["insert_predecessor"]

            current_deps = get_dependencies(dsk_dict, k, as_list=True)
            k_in_keys = None
            if len(dsk_dict[k]) > DATA:
                k_in_keys = deepcopy(dsk_dict[k][DATA])  # [DATA] equals in_keys in dict

            if not isinstance(new_request, list):
                clone_dependencies = [new_request]
            else:
                clone_dependencies = new_request

            # check if any of our current dependencies already has to fulfil a request
            # since the request's might collide we should just duplicate it
            # in this run it gets a new name, and the existing one is left untouched until it's its turn.
            clone = False
            if clone_instead_merge:
                if len(clone_dependencies) > 1:
                    clone = True
                    k_in_keys = []
                else:
                    for d in current_deps:
                        if len(requests.get(d, [])) > 0:
                            clone = True
                            k_in_keys = []

            # if it's a list it automatically clones it, if its not
            # the user could use the clone_dependencies to clone it
            if (
                new_request is not None
                and not isinstance(new_request, list)
                and "clone_dependencies" in new_request
                and new_request["clone_dependencies"]
            ):
                clone_dependencies = new_request["clone_dependencies"]
                del new_request["clone_dependencies"]
                clone = True
                k_in_keys = []

            if (
                new_request is not None
                and not isinstance(new_request, list)
                and "requires_request" in new_request
                and new_request["requires_request"]
            ):
                del new_request["requires_request"]
                # input_requests[k] = NestedFrozenDict(new_request[0])
                input_requests[k] = new_request["self"]

            # all_deps = get_all_dependencies()
            clone_dependencies = [c for c in clone_dependencies if c is not None]

            for clone_id, request in enumerate(clone_dependencies):
                if clone:
                    to_clone_keys = dsk_dict[k][DATA]
                    if not isinstance(to_clone_keys, list):
                        to_clone_keys = [to_clone_keys]

                    # create new node in graph containing k_in_keys as input
                    if insert_predecessor:
                        pre_function = insert_predecessor[clone_id]
                        pre_request = clone_dependencies[clone_id]

                        pre_k = tokenize([k, "deklare_pre", clone_id])
                        pre_base_name = None
                        if hasattr(pre_function, "__self__") and hasattr(
                            pre_function.__self__, "dask_key_name"
                        ):
                            pre_base_name = pre_function.__self__.dask_key_name
                            pre_k = pre_base_name + KEY_SEP + pre_k

                        # # go trough request and update the name of the clone
                        # if pre_base_name is not None:
                        #     update_key_in_config(pre_request,pre_base_name,pre_k)

                        requests[pre_k] = [pre_request]
                        dsk_dict[pre_k] = [apply, pre_function, [], {}]
                        pre_in_keys = []

                    for to_clone_key in to_clone_keys:
                        if insert_predecessor:
                            if to_clone_key is None:
                                pre_in_keys.append(None)
                            else:
                                pre_in_keys.append(
                                    generate_clone_key(k, to_clone_key, clone_id)
                                )
                        else:
                            if to_clone_key is None:
                                k_in_keys.append(None)
                            else:
                                k_in_keys.append(
                                    generate_clone_key(k, to_clone_key, clone_id)
                                )

                    if insert_predecessor:
                        dsk_dict[pre_k][DATA] = pre_in_keys
                        dsk_dict[pre_k] = tuple(dsk_dict[pre_k])

                        k_in_keys += [pre_k]
                        remove[pre_k] = False
                        new_work[pre_k] = True
                        # new_work.append(pre_k)

                for i, d in enumerate(current_deps):
                    # duplicate keys

                    if clone:
                        clone_work = [d]

                        d = generate_clone_key(k, d, clone_id)
                        while clone_work:
                            new_clone_work = []
                            for cd in clone_work:
                                clone_d = generate_clone_key(k, cd, clone_id)

                                # update_key_in_config(request,cd,clone_d)
                                # TODO: do we need to reset the dask_key_name of each
                                #       of each cloned node?

                                normalize_node(cd)

                                cloned_cd_node = copy(dsk_dict[cd])

                                # if contains data as input
                                to_clone_keys = cloned_cd_node[DATA]
                                if not isinstance(to_clone_keys, list):
                                    to_clone_keys = [to_clone_keys]
                                cd_in_keys = []
                                for to_clone_key in to_clone_keys:
                                    if to_clone_key is None:
                                        cd_in_keys.append(None)
                                    else:
                                        cd_in_keys.append(
                                            generate_clone_key(
                                                k, to_clone_key, clone_id
                                            )
                                        )
                                # if len(cd_in_keys) == 1:
                                #     cd_in_keys = cd_in_keys[0]
                                nd = list(cloned_cd_node)
                                nd[DATA] = cd_in_keys
                                cloned_cd_node = tuple(nd)
                                dsk_dict[clone_d] = cloned_cd_node
                                new_deps = get_dependencies(dsk_dict, cd, as_list=True)
                                new_clone_work += new_deps
                            clone_work = new_clone_work

                    # determine what needs to be removed
                    if not insert_predecessor:
                        # we are not going to remove anything if we inserted a predecessor node before current node k
                        # we are also not updating the requests of dependencies of the original node k
                        # since it will be done in the next interaction by configuring the inserted predecessor

                        to_be_removed = False
                        if k in remove:
                            to_be_removed = remove[k]
                        if request is None:
                            to_be_removed = True
                        if "remove_dependencies" in request:
                            to_be_removed = request["remove_dependencies"]
                            del request["remove_dependencies"]

                        # TODO: so far this doesn't allow to clone dependencies and delete only one of them.
                        #       it might be irrelevant.
                        if request.get("remove_dependency", {}).get(
                            base_name(d), False
                        ):
                            to_be_removed = True
                            del request["remove_dependency"][base_name(d)]

                        if not request.get("remove_dependency", True):
                            # clean up if an empty dict still exists
                            del request["remove_dependency"]
                        if d in requests:
                            if clone_instead_merge:
                                raise InternalError(
                                    f"A duplicate request was found for {d} with the request {requests[d]}, set clone_instead_merge=False to allow this"
                                )
                            if not to_be_removed:
                                requests[d] += [request]
                            remove[d] = remove[d] and to_be_removed
                        else:
                            if not to_be_removed:
                                requests[d] = [request]
                            # if we received None
                            remove[d] = to_be_removed

                        # only configure each node once in a round!
                        # if d not in new_work and d not in work:
                        #     new_work.append(
                        #         d
                        #     )  # TODO: Do we need to configure dependency if we'll remove it?
                        # we should also add `d`` only to the work list if we did not insert a
                        # a predecessor. The predecessor will take care of adding it in the next round
                        # otherwise it could happen that the predecessor changes the name of the node
                        # by cloning it. Then we'd have a deprecated node name in the work list
                        if d not in work and (d not in remove or not remove[d]):
                            new_work[d] = True
            dsk_k = list(dsk_dict[k])
            # loop though all input keys of this node `k` and discard all inputs that have been removed
            if k_in_keys is not None:
                if not isinstance(k_in_keys, list):
                    k_in_keys = [k_in_keys]
                k_in_keys = [
                    k_in_keys_k
                    for k_in_keys_k in k_in_keys
                    if (not remove.get(k_in_keys_k, False))
                ]
                # if len(k_in_keys) == 1:
                #     k_in_keys = k_in_keys[0]
                dsk_k[DATA] = k_in_keys
            dsk_dict[k] = tuple(dsk_k)

        work = new_work

    # Assembling the configured new graph
    out = {k: dsk_dict[k] for k in out_keys if not remove[k]}

    def clean_request(x):
        """Removes any leftovers of deklare configuration"""
        for item in ["remove_dependencies", "clone_dependencies", "insert_predecessor"]:
            x.pop(item, None)

        return x

    # After we have acquired all requests we can input the required_requests as a input node to the requiring node
    # we assume that the last argument is the request
    for k in input_requests:
        if k not in out:
            continue
        # input_requests[k] = clean_request(input_requests[k])
        # Here we assume that we always receive the same tuple of (bound method, data, request)
        # If the interface changes this will break #TODO: check for all cases
        if isinstance(out[k][REQUEST], tuple):
            # FIXME: find a better inversion of unpack_collections().
            #        this is very fragile
            # Check if we've already got a request as argument
            # This is the case if our node will make use of a general config
            # Then the present request is updated with the configured one
            # We need to recreate the tuple/list elements though. (dask changed)
            # TODO: use a distinct request class
            if out[k][REQUEST][0] is dict:
                my_dict = {}
                # FIXME: it does not account for nested structures
                for item in out[k][REQUEST][1]:
                    if isinstance(item[1], tuple):
                        if item[1][0] is tuple:
                            item[1] = tuple(item[1][1])
                        elif item[1][0] is list:
                            item[1] = list(item[1][1])
                    my_dict[item[0]] = item[1]
                my_dict = {item[0]: item[1] for item in out[k][REQUEST][1]}
                my_dict.update(input_requests[k])
                out[k] = out[k][:REQUEST] + (my_dict,)
            else:
                # replace the last entry
                out[k] = out[k][:REQUEST] + (input_requests[k],)

        # # TODO: verify that we can ignore this case
        # elif isinstance(out[k][REQUEST], dict):
        #     out[k] = out[k][:REQUEST] + (copy(out[k][REQUEST]) | copy(input_requests[k]),)
        else:
            # replace the last entry
            out[k] = out[k][:REQUEST] + (input_requests[k],)

        # TODO: we might dask.delayed(out[k][REQUEST]) here

    # convert to delayed object
    in_keys = list(flatten(keys))

    def dask_optimize(dsk, keys):
        dsk1, deps = cull(dsk, keys)
        dsk2 = inline(dsk1, dependencies=deps)
        # dsk3 = inline_functions(dsk2, keys, [len, str.split],
        #                        dependencies=deps)
        dsk4, deps = fuse(dsk2)
        return dsk4

    #    dsk, dsk_keys = dask.base._extract_graph_and_keys([collection])

    # out = optimize_functions(out, in_keys)
    #    collection = Delayed(key=dsk_keys, dsk=collection)

    if len(in_keys) > 1:
        # collection = [Delayed(key=key, dsk=out) for key in in_keys]
        collection = Delayed(key=in_keys, dsk=out)
    else:
        collection = Delayed(key=in_keys[0], dsk=out)
        if isinstance(delayed, list):
            collection = [collection]

    # if isinstance(delayed, list):
    #     collection = [collection]

    if optimize_graph:
        collection = optimize(collection, keys)
    #
    return collection


def dfs_order(dsk, keys=None, dask_optimize=False):
    """performs a depth first search and orders the items accordingly"""

    # if not isinstance(delayed, list):
    #     collections = [delayed]
    # else:
    #     collections = delayed

    # dsk, dsk_keys = dask.base._extract_graph_and_keys(collections)

    # dsk_dict = {k: dsk[k] for k in dsk.keys()}
    dsk_dict = dsk

    dependencies = {k: get_dependencies(dsk, k) for k in dsk}
    dependents = reverse_dict(dependencies)
    root_nodes = {k for k, v in dependents.items() if not v}
    keys = root_nodes

    if not isinstance(keys, (list, set)):
        keys = [keys]

    work = list(set(flatten(keys)))

    def dfs(k, dsk_dict, keyorder, idx):
        current_deps = get_dependencies(dsk_dict, k, as_list=True)
        for i, d in enumerate(current_deps):
            if d not in keyorder:
                keyorder, idx = dfs(d, dsk_dict, keyorder, idx)
                keyorder[d] = idx
                idx = idx + 1

        return keyorder, idx

    keyorder = {}
    idx = 0
    for k in work:
        keyorder, idx = dfs(k, dsk_dict, keyorder, idx)
        if k not in keyorder:
            keyorder[k] = idx
            idx += 1

    return keyorder


def optimize(delayed, keys=None, dask_optimize=False):
    """Optimizes the graph after configuration"""

    # TODO: Why is this necessary?
    if not isinstance(delayed, list):
        collections = [delayed]
    else:
        collections = delayed

    dsk, dsk_keys = dask.base._extract_graph_and_keys(collections)

    dsk_dict = {k: dsk[k] for k in dsk.keys()}

    if keys is None:
        keys = dsk_keys

    if not isinstance(keys, (list, set)):
        keys = [keys]

    work = list(set(flatten(keys)))

    # Invert the task graph: make a compute graph
    dsk_inv = {k: {} for k in work}

    out_keys = []
    sources = set()
    seen = set()
    while work:
        new_work = []

        out_keys += work
        for k in work:
            current_deps = get_dependencies(dsk_dict, k, as_list=True)

            if not current_deps:
                sources.add(k)

            for i, d in enumerate(current_deps):
                if d not in dsk_inv:
                    dsk_inv[d] = {k}
                else:
                    dsk_inv[d].add(k)

                if d not in seen:
                    new_work.append(d)
                    seen.add(d)

        work = new_work

    def replace(s, r, n):
        if isinstance(s, str) and s == r:
            return n
        return s

    # traverse the task graph in compute direction

    # starting from the sources
    work = list(sources)
    in_keys = list(set(flatten(keys)))
    out_keys = []
    seen = set()
    # import time
    # debug={}
    rename = {}
    while work:
        new_work = []

        for k in work:
            if k in in_keys:
                # if we are at a sink node we don't change the name
                out_keys += [k]
                continue

            # rename k
            node_token = ""

            # FIXME: for which cases do we need the following two lines?
            # it breaks the optimization significantly for chunkpersister, because it initializes
            # a new hashpersister instance for each parallel branch dynamically
            # What is the purpose? Do we want to account for internal configurations? Maybe find a different solution
            # if hasattr(dsk_dict[k][0], "__self__"):
            #     node_token = dsk_dict[k][0].__self__

            input_token = list(dsk_dict[k][1:])

            token = tokenize([node_token, input_token])

            # start = time.time()
            new_k = base_name(k) + KEY_SEP + token
            # duration = time.time() - start
            # debug[base_name(k)] = debug.get(base_name(k),0) + duration

            out_keys += [new_k]
            rename[k] = new_k

            current_dependants = [d for d in dsk_inv[k]]

            for i, d in enumerate(current_dependants):
                # TODO: is there a way to not change the dsk_dict in-place?
                if isinstance(dsk_dict[d][DATA], list):
                    in1 = [replace(s, k, new_k) for s in dsk_dict[d][DATA]]
                elif isinstance(dsk_dict[d][DATA], str):
                    in1 = replace(dsk_dict[d][DATA], k, new_k)
                else:
                    in1 = dsk_dict[d][DATA]

                dsk_dict[d] = tuple(
                    list(dsk_dict[d][:DATA]) + [in1] + list(dsk_dict[d][3:])
                )

                if d not in seen:
                    new_work.append(d)
                    seen.add(d)

        work = new_work

    for k in rename:
        dsk_dict[rename[k]] = dsk_dict[k]

    out = {k: dsk_dict[k] for k in out_keys}

    def optimize_functions(dsk, keys):
        dsk1, deps = cull(dsk, keys)
        dsk2 = inline(dsk1, dependencies=deps)
        # dsk3 = inline_functions(dsk2, keys, [len, str.split],
        #                        dependencies=deps)
        dsk4, deps = fuse(dsk2, fuse_subgraphs=True)
        return dsk4

    if dask_optimize:
        out = optimize_functions(out, in_keys)

    if len(in_keys) > 1:
        # collection = [Delayed(key=key, dsk=out) for key in in_keys]
        collection = Delayed(key=in_keys, dsk=out)
    else:
        collection = Delayed(key=in_keys[0], dsk=out)
        if isinstance(delayed, list):
            collection = [collection]

    return collection


def compute(graph, request):
    configured_graph = configuration(graph, request)
    # dsk, dsk_keys = dask.base._extract_graph_and_keys([configured_graph])

    computed_result = dask.compute(configured_graph)[0]
    return computed_result


class create_taskgraph:
    def __init__(self):
        self.prev = False
        pass

    def __enter__(self):
        if setting_exists("use_delayed"):
            self.prev = get_setting("use_delayed")
        set_setting("use_delayed", True)

    def __exit__(self, type, value, traceback):
        set_setting("use_delayed", self.prev)
