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
from types import TracebackType
from typing import Any, Callable
from uuid import uuid4

import dask
import dask.delayed
from dask.delayed import Delayed

KEY_SEP = "+"
PROTECTED_DESKRIPTOR_KEYS = ["self", "config"]
PROTECTED_CONFIG_KEYS = ["global", "types", "keys"]

# ToDo: Doc strings
# ToDo: more precise types


class FlowContext:
    __context: dict[str, Callable] = {}
    __enabled: bool = False

    @classmethod
    def get(cls, name: str) -> Callable:
        return cls.__context[name]

    @staticmethod
    def exists(name: str) -> bool:
        return name in FlowContext.__context

    @staticmethod
    def set(name: str, value: Callable) -> None:
        FlowContext.__context[name] = value

    @staticmethod
    def is_enabled() -> bool:
        return FlowContext.__enabled

    @staticmethod
    def set_enabled(value: bool) -> None:
        FlowContext.__enabled = value

    @staticmethod
    def reset() -> None:
        FlowContext.__enabled = False
        FlowContext.__context = {}


class TaskGraphCreator:
    prev: bool

    def __init__(self) -> None:
        self.prev = False

    def __enter__(self) -> None:
        global is_enabled
        FlowContext.reset()
        FlowContext.set_enabled(True)
        is_enabled = True

    def __exit__(
        self, _type: BaseException | None, _value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        global is_enabled
        is_enabled = False
        FlowContext.reset()


is_enabled = False


class Node(object):
    _name: str | None
    config: dict

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
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

    def merge_config(self, deskriptor: dict) -> dict:
        """Each deskriptor contains configuration which may apply to different
        node instances. This function collects all information that apply to _this_
        node (including it's preset configs) and adds a `self` keyword to the deskriptor.

        Args:
            deskriptor (dict): The deskriptor and configuration options.

        Returns:
            dict: A new deskriptor which specific to this node.
        """
        new_deskriptor = self._copy_deskriptor(deskriptor)

        self._update_deskriptor_config(deskriptor, new_deskriptor)

        return new_deskriptor

    def configure(self, deskriptor: dict) -> dict:
        """Before a task graph is executed each node is configured.
            The deskriptor is propagated from the end to the beginning
            of the DAG and each nodes "configure" routine is called.
            The deskriptor can be updated to reflect additional requirements,
            The return value gets passed to predecessors.

            Essentially the following question must be answered within the
            nodes configure function:
            What do I need to fulfil the deskriptor of my successor? Either the node
            can provide what is required or the deskriptor is passed through to
            predecessors in hope they can fulfil the deskriptor.

            Here, you must not configure the internal parameters of the
            Node otherwise it would not be thread-safe. You can however
            introduce a new key 'requires_deskriptor' in the deskriptor being
            returned. This deskriptor will then be passed as an argument
            to the __call__ function.

            Best practice is to configure the Node on initialization with
            runtime independent configurations and define all runtime
            dependant configurations here.

        Args:
            deskriptor (dict): deskriptor to merge with own config.


        Returns:
            dict -- The (updated) deskriptor. If updated, modifications
                    must be made on a copy of the input. The return value
                    must be a dictionary.
                    If multiple deskriptors are input to this function they
                    must be merged.
                    If nothing needs to be deskriptored an empty dictionary
                    can be return. This removes all dependencies of this
                    node from the task graph.

        """
        merged_deskriptor = self.merge_config(deskriptor)

        # set default
        merged_deskriptor["requires_deskriptor"] = True

        return merged_deskriptor

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        name = kwargs.get("name", None)
        context = kwargs.get("context", None)
        if name is not None and KEY_SEP in name:
            raise RuntimeError(f"Do not use a `{KEY_SEP}` character in your {name=}")
        if name is None:
            name = self._name

        new_kwargs = copy(kwargs)
        if kwargs.get("deskriptor", None) is not None:
            new_kwargs["deskriptor"] = self.merge_config(kwargs["deskriptor"])
        elif kwargs.get("deskriptor", None) is not None:
            new_kwargs["deskriptor"] = self.merge_config(kwargs["deskriptor"])
        else:
            new_kwargs["deskriptor"] = self.merge_config({})

        if context is None:
            context = FlowContext

        forward_func = self.compute

        if context.is_enabled():
            func = dask.delayed(forward_func)(*args, dask_key_name=name, **kwargs)
            self.dask_key_name = func.key
            return func
        else:
            return forward_func(*args, **kwargs)

    def _copy_deskriptor(self, deskriptor: dict) -> dict:
        new_deskriptor = deepcopy(deskriptor)

        new_deskriptor["self"] = {}
        if hasattr(self, "config"):
            new_deskriptor["self"].update(deepcopy(self.config))

        for key in deskriptor:
            if key not in PROTECTED_DESKRIPTOR_KEYS:
                new_deskriptor["self"][key] = deskriptor[key]

        return new_deskriptor

    def _update_deskriptor_config(self, old_deskriptor: dict, new_deskriptor: dict) -> None:
        assert isinstance(new_deskriptor["self"], dict)  # for type hints, is set to dict in _copy_deskriptor

        if old_deskriptor.get("config", None) is not None:
            # go through all parameters in the deskriptor's config and add them to the self parameters

            # assume anything within 'config' is global
            for key in old_deskriptor["config"]:
                if key in PROTECTED_CONFIG_KEYS:
                    continue

                new_deskriptor["self"][key] = old_deskriptor["config"][key]

            # add specific global config entries
            if "global" in old_deskriptor["config"]:
                for key in old_deskriptor["config"]["global"]:
                    new_deskriptor["self"][key] = old_deskriptor["config"]["global"][key]

            # add type specific configs (overwrites global config)
            if "types" in old_deskriptor["config"]:
                if type(self).__name__ in old_deskriptor["config"]["types"]:
                    new_deskriptor["self"].update(deepcopy(old_deskriptor["config"]["types"][type(self).__name__]))

            # add key specific configs (overwrites global and type config)
            if "keys" in old_deskriptor["config"]:
                if self.dask_key_name in old_deskriptor["config"]["keys"]:
                    new_deskriptor["self"].update(deepcopy(old_deskriptor["config"]["types"][self.dask_key_name]))

                    # TODO: It should be safe to remove these keys from the new_deskriptor!?
                    del new_deskriptor["config"]["keys"][self.dask_key_name]

                # TODO: should we prefer the following way of removing the config?
                # new_deskriptor['config']['keys'] = {k:v for k,v in old_deskriptor["config"]["keys"].items() if k != self.dask_key_name}  # noqa: E501

            new_deskriptor["config"] = old_deskriptor["config"]


# ToDo: fix types
def init_flow_graph(flow: type | Callable):
    if inspect.isclass(flow):
        flow = flow()

    with TaskGraphCreator():
        flow_graph = flow()

    return flow_graph


def task(name: str | None = None, context: FlowContext | None = None) -> Callable:
    context = context or FlowContext

    def decorator_task(func_or_cls: type | Callable) -> type | Callable:
        return (
            _wrap_class(func_or_cls, name)
            if inspect.isclass(func_or_cls)
            else _wrap_function(func_or_cls, name, context)
        )

    return decorator_task


def _wrap_class(cls: type, name: str | None = None) -> type:
    if cls.__name__ == "DeklareClass":  # Keep this check if "DeklareClass" is still a sentinel
        # don't wrap it twice!
        return cls

    # Create a new class dynamically with the original class's name
    # The new class inherits from the original cls and Node
    new_cls_name = cls.__name__
    bases = (cls, Node)
    new_cls_dict = {}

    # Define __init__ for the new class
    def new_init(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN001, ANN401
        super(type(self), self).__init__(*args, **kwargs)
        self._name = getattr(self, "_name", None) or name

    new_cls_dict["__init__"] = new_init

    # Dynamically create the new class
    new_cls = type(new_cls_name, bases, new_cls_dict)

    # Check if the original class defines configure
    if "configure" in cls.__dict__:
        original_inherit_method = cls.__dict__["configure"]

        def new_configure(self: type, deskriptor: dict) -> dict:
            # Ensure Node.configure is called correctly
            deskriptor = Node.configure(self, deskriptor)
            return original_inherit_method(self, deskriptor)

        new_cls.configure = new_configure

    # Rename cls's __call__ method to compute
    if "__call__" in cls.__dict__:
        # Directly set 'compute' to the original __call__ method
        new_cls.compute = cls.__dict__["__call__"]
        # Use Node's __call__ method as NewWrappedClass's __call__ method
        new_cls.__call__ = Node.__call__

    return new_cls


def _wrap_function(func: Callable, name: str | None, context: FlowContext) -> Callable:
    if isinstance(func, Delayed):
        # don't wrap it twice!
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
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
                warnings.warn(f"Duplicate name detected. Name changed to {key_name}", stacklevel=1)

            # make a graph node
            return dask.delayed(func)(*args, dask_key_name=key_name, **kwargs)
        else:
            # compute function and return result
            return func(*args, **kwargs)

    return wrapper
