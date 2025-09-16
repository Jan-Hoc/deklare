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

import importlib
import inspect
import yaml
import os
from pathlib import Path
from typing import Callable

from .core import init_flow_graph, task
from .deskribe import Deskriptor
from .graph import Node, compute
from .persist import ChunkPersister, Persister

# ToDo: Doc strings
# ToDo: more precise types


def deklare_flow(
    flow,
    template_deskriptor: Deskriptor | None = None,
    config_path: Path | str | None = None,
):
    flow_graph = init_flow_graph(flow)

    config_deskriptor = None
    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_deskriptor = yaml.safe_load(f)

    def deklare_flow_function(self, deskriptor):
        return compute(flow_graph, deskriptor)

    def query_class(self, deskriptor):
        return query(deskriptor)

    def query(deskriptor):
        if config_deskriptor:
            deskriptor = Deskriptor.update_from_config_dict(deskriptor, config_deskriptor)

        if template_deskriptor:
            deskriptor = template_deskriptor.from_dict(deskriptor).to_dict()
        return compute(flow_graph, deskriptor)

    if hasattr(flow, "__self__") and flow.__self__ is not None:
        flow.query = query_class
    else:
        flow.query = query

    return flow


def is_module_function_or_class(_module: importlib.ModuleType) -> Callable:
    def predicate(member: object) -> bool:
        # TODO: should we also check for member.__name__ in module.__name__?
        return inspect.isfunction(member) or inspect.isclass(member)

    return predicate


def deklare_module(
    module: str | importlib.ModuleType,
    flows: str | set | None = None,
    names: dict | None = None,
    external_tasks: list | None = None,
    ignore: list | None = None,
    no_wrap: bool = False,
) -> tuple:
    names = names or {}
    external_tasks = external_tasks or []
    ignore = ignore or []

    flows = flows or set()
    if isinstance(flows, str):
        flows = {flows}

    flows = set(flows)

    if isinstance(module, str):
        # Dynamically import the module
        module = importlib.import_module(module)

    flow_instances, dependency_modules = _init_modules(module, flows, ignore, names, external_tasks)

    for dependency_module, dependency_flows in dependency_modules.items():
        dependency_flows_names = list(dependency_flows.keys())
        module_and_flows = deklare_module(dependency_module, flows=dependency_flows_names, no_wrap=True)
        for _, flow in enumerate(module_and_flows[1:]):
            setattr(module, flow.__name__, flow)
            # # TODO: would the above fail if we use `from module import flow as f`?
            # # maybe better to do, but probably not always in the same order!
            # setattr(module, dependency_flows[i], flow)

    for name, func_or_cls in flow_instances.items():
        if no_wrap:
            flow_instances[name] = func_or_cls
        else:
            flow_instances[name] = deklare_flow(func_or_cls)

    return tuple([module] + list(flow_instances.values()))


def _init_modules(module: importlib.ModuleType, flows: set, ignore: list, names: dict, external_tasks: list) -> tuple:
    flow_instances, dependency_modules = {}, {}

    # Iterate over all functions defined in the module
    for name, func_or_cls in inspect.getmembers(module, predicate=is_module_function_or_class(module)):
        if name in flows:
            # if member direct member of the module, we add it to the return flows
            # if not, it needs to be loaded appropriately from it's original module
            if func_or_cls.__module__ == module.__name__:
                flow_instances[name] = func_or_cls
            else:
                # select to load it as a deklare dependency, i.e. load all required members
                # as deklare tasks
                dependency_modules[func_or_cls.__module__] = dependency_modules.get(func_or_cls.__module__, {}) | {
                    name: func_or_cls
                }
        elif name not in ignore and (func_or_cls.__module__ == module.__name__ or name in external_tasks):
            # Decorate the function and add it back to the module's namespace
            key_names = names.get(name, None)
            setattr(module, name, task(name=key_names)(func_or_cls))

    return flow_instances, dependency_modules
