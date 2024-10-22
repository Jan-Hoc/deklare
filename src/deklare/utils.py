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

import dask
from dask.delayed import Delayed

from .graph import base_name


def indexers_to_slices(indexers):
    new_indexers = {}
    for key in indexers:
        if isinstance(indexers[key], dict):
            ni = {"start": None, "end": None, "step": None}
            ni.update(indexers[key])
            new_indexers[key] = slice(ni["start"], ni["end"], ni["step"])
        else:
            new_indexers[key] = indexers[key]

    return new_indexers


def exclusive_indexing(x, indexers):
    # Fake `exlusive indexing`
    drop_indexers = {k: indexers[k]["end"] for k in indexers if "end" in indexers[k]}
    try:
        x = x.drop_sel(drop_indexers, errors="ignore")
    except Exception:
        pass

    return x


class NodeFailedException(Exception):
    def __init__(self, exception=None):
        """The default exception when a node's compute function fails and failsafe mode
        is enable, i.e. the global setting `fail_mode` is not set to `fail`.
        this exception is caught by the foreal processing system and depending on the
        global variable `fail_mode`, leads to process interruption or continuation.

        Args:
            exception (any, optional): The reason why it failed, e.g. another exception.
                Defaults to None.
        """
        # if get_setting("fail_mode") == "warning" or get_setting("fail_mode") == "warn":
        #     print(exception)
        self.exception = exception

    def __str__(self):
        return str(self.exception)


def dict_update(base, update):
    if not isinstance(base, dict) or not isinstance(update, dict):
        raise TypeError(
            f"dict_update requires two dicts as input. But we received {type(base)} and {type(update)}"
        )

    for key in update:
        if isinstance(base.get(key), dict) and isinstance(update[key], dict):
            base[key] = dict_update(base[key], update[key])
        else:
            base[key] = update[key]

    return base


def extract_subgraphs(taskgraph, keys, match_base_name=False):
    if not isinstance(taskgraph, list):
        taskgraph = [taskgraph]

    extracted_graph, ck = dask.base._extract_graph_and_keys(taskgraph)
    if match_base_name:
        configured_graph_keys = list(extracted_graph.keys())
        new_keys = []
        for k in configured_graph_keys:
            for sk in keys:
                if base_name(sk) == base_name(k):
                    new_keys += [k]
        keys = new_keys
    return Delayed(keys, extracted_graph)
