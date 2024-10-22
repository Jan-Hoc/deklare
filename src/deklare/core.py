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

import functools
import inspect
import warnings
from copy import copy, deepcopy
from uuid import uuid4

import dask
import dask.delayed
from dask.delayed import Delayed

KEY_SEP = "+"


class FlowContext:
    __context: dict = {}
    __enabled: bool = False

    @classmethod
    def get(name):
        return FlowContext.__context[name]

    @staticmethod
    def exists(name):
        return name in FlowContext.__context

    @staticmethod
    def set(name, value):
        FlowContext.__context[name] = value

    @staticmethod
    def is_enabled():
        return FlowContext.__enabled

    @staticmethod
    def set_enabled(value: bool):
        FlowContext.__enabled = value

    @staticmethod
    def reset():
        FlowContext.__enabled = False
        FlowContext.__context = {}


class create_taskgraph:
    def __init__(self):
        self.prev = False

    def __enter__(self):
        global is_enabled
        FlowContext.reset()
        FlowContext.set_enabled(True)
        is_enabled = True

    def __exit__(self, type, value, traceback):
        global is_enabled
        is_enabled = False
        FlowContext.reset()


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


is_enabled = False


class Node(object):
    def __init__(self, **kwargs):
        self.config = locals().copy()
        # FIXME: This works but there are better solutions!
        while "kwargs" in self.config:
            if "kwargs" not in self.config["kwargs"]:
                self.config.update(self.config["kwargs"])
                break
            self.config.update(self.config["kwargs"])

        del self.config["kwargs"]
        del self.config["self"]

        self._name = None

    def merge_config(self, request):
        """Each request contains configuration which may apply to different
        node instances. This function collects all information that apply to _this_
        node (including it's preset configs) and adds a `self` keyword to the request.

        Args:
            request (dict): The request and configuration options.

        Returns:
            dict: A new request which specific to this node.
        """
        protected_keys = ["self", "config"]

        new_request = {}
        new_request.update(request)
        new_request["self"] = {}
        if hasattr(self, "config"):
            new_request["self"].update(deepcopy(self.config))
        for key in request:
            # we consider any key that is not a protected key a global key
            if key not in protected_keys:
                new_request["self"][key] = request[key]

        if "config" in request:
            PROTECTED_CONFIG_KEYS = ["global", "types", "keys"]

            # new_request['config'] = {}

            # We'll go through all parameters in the request's config
            # and add them to the self parameters

            # We assume that anything within 'config' is global
            for key in request["config"]:
                if key in PROTECTED_CONFIG_KEYS:
                    continue
                new_request["self"][key] = request["config"][key]

                # # add to new_request
                # new_request['config'][key] = request["config"][key]

            # add specific global config entries
            if "global" in request["config"]:
                for key in request["config"]["global"]:
                    new_request["self"][key] = request["config"]["global"][key]

                # # add to new_request
                # new_request['config']['global'] = request["config"]["global"]

            # add type specific configs (overwrites global config)
            if "types" in request["config"]:
                if type(self).__name__ in request["config"]["types"]:
                    dict_update(
                        new_request["self"],
                        request["config"]["types"][type(self).__name__],
                    )

                # # add to new_request
                # new_request['config']['types'] = request["config"]["types"]

            # add key specific configs (overwrites global and type config)
            if "keys" in request["config"]:
                if self.dask_key_name in request["config"]["keys"]:
                    dict_update(
                        new_request["self"],
                        request["config"]["keys"][self.dask_key_name],
                    )

                    # TODO: It should be save to remove these keys from the new_request!?
                    del new_request["config"]["keys"][self.dask_key_name]

                # TODO: should we prefer the following way of removing the config?
                # new_request['config']['keys'] = {k:v for k,v in request["config"]["keys"].items() if k != self.dask_key_name}

        if "config" in request:
            new_request["config"] = request["config"]
        return new_request

    def configure(self, request):
        """Before a task graph is executed each node is configured.
            The request is propagated from the end to the beginning
            of the DAG and each nodes "configure" routine is called.
            The request can be updated to reflect additional requirements,
            The return value gets passed to predecessors.

            Essentially the following question must be answered within the
            nodes configure function:
            What do I need to fulfil the request of my successor? Either the node
            can provide what is required or the request is passed through to
            predecessors in hope they can fulfil the request.

            Here, you must not configure the internal parameters of the
            Node otherwise it would not be thread-safe. You can however
            introduce a new key 'requires_request' in the request being
            returned. This request will then be passed as an argument
            to the __call__ function.

            Best practice is to configure the Node on initialization with
            runtime independent configurations and define all runtime
            dependant configurations here.

        Args:
            requests {List} -- List of requests (i.e. dictionaries).


        Returns:
            dict -- The (updated) request. If updated, modifications
                    must be made on a copy of the input. The return value
                    must be a dictionary.
                    If multiple requests are input to this function they
                    must be merged.
                    If nothing needs to be requested an empty dictionary
                    can be return. This removes all dependencies of this
                    node from the task graph.

        """
        merged_request = self.merge_config(request)

        # set default
        merged_request["requires_request"] = True
        return merged_request

    def __call__(self, *args, **kwargs):
        name = kwargs.get("name", None)
        context = kwargs.get("context", None)
        if name is not None and KEY_SEP in name:
            raise RuntimeError(f"Do not use a `{KEY_SEP}` character in your {name=}")
        if name is None:
            name = self._name

        new_kwargs = copy(kwargs)
        if kwargs.get("request", None) is not None:
            new_kwargs["request"] = self.merge_config(kwargs["request"])
        elif kwargs.get("deskriptor", None) is not None:
            new_kwargs["deskriptor"] = self.merge_config(kwargs["deskriptor"])
        else:
            new_kwargs["request"] = self.merge_config({})

        if context is None:
            context = FlowContext

        forward_func = self.compute

        if context.is_enabled():
            func = dask.delayed(forward_func)(*args, dask_key_name=name, **kwargs)
            # func = dask.delayed(forward_func)(data=None, request=None,
            #     dask_key_name=name
            # )
            self.dask_key_name = func.key
            return func
        else:
            return forward_func(*args, **kwargs)


def it(flow):
    if inspect.isclass(flow):
        flow = flow()

    with create_taskgraph():
        flow_graph = flow()

    return flow_graph


def task(name=None, context=None):
    if context is None:
        context = FlowContext

    def decorator_task(func_or_cls):
        if inspect.isclass(func_or_cls):
            cls = func_or_cls

            if cls.__name__ == "DeklareClass":
                # don't wrap it twice!
                return cls

            class DeklareClass(cls, Node):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._name = name

            # Check if the original class defines configure
            if "configure" in cls.__dict__:
                original_inherit_method = cls.__dict__["configure"]

                def new_configure(self, request):
                    request = Node.configure(self, request)
                    return original_inherit_method(self, request)

                setattr(DeklareClass, "configure", new_configure)

            # Rename cls's __call__ method to compute
            if "__call__" in cls.__dict__:
                setattr(DeklareClass, "compute", cls.__dict__["__call__"])
                # Use Node's __call__ method as NewClass's __call__ method
                setattr(DeklareClass, "__call__", Node.__call__)

            return DeklareClass
        else:
            func = func_or_cls

            if isinstance(func, Delayed):
                # don't wrap it twice!
                return func

            @functools.wraps(func)
            def wrapper_decorator(*args, **kwargs):
                if context.is_enabled():
                    if name is None:
                        key_name = func.__name__
                    else:
                        key_name = name

                    ext = ""
                    while context.exists(key_name + ext):
                        ext = uuid4().hex[-6:]

                    key_name = key_name + ext
                    context.set(key_name, func)

                    if ext != "":
                        warnings.warn(
                            f"Duplicate name detected. Name changed to {key_name}"
                        )

                    # make a graph node
                    return dask.delayed(func)(*args, dask_key_name=key_name, **kwargs)
                else:
                    # compute function and return result
                    return func(*args, **kwargs)

        return wrapper_decorator

    return decorator_task
