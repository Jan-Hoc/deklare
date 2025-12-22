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

import itertools
import math
import warnings
from typing import Any, Iterable, get_args, get_origin


import pandas as pd
import xarray as xr
from dask.base import _extract_graph_and_keys
from dask.delayed import Delayed
from dask.typing import Graph
from pandas._libs.tslibs.nattype import NaTType
from pandas.core.tools.datetimes import DatetimeScalar

from .core import KEY_SEP
from .descriptor import DatetimeRange, Descriptor, Range

# ToDo: Doc strings


def base_name(name: str) -> str:
    return name.split(KEY_SEP)[0]


def indexers_to_slices(indexers: dict) -> dict:
    new_indexers = {}
    for idxr, key in indexers.items():
        if isinstance(idxr, dict):
            new_idxr = {"start": None, "end": None, "step": None}
            new_idxr.update(idxr)
            new_indexers[key] = slice(new_idxr["start"], new_idxr["end"], new_idxr["step"])
        else:
            new_indexers[key] = idxr

    return new_indexers


def exclusive_indexing(x: xr.DataArray, indexers: dict) -> xr.DataArray:
    # Fake `exlusive indexing`
    drop_indexers = {k: indexers[k]["end"] for k in indexers if "end" in indexers[k]}
    try:
        x = x.drop_sel(drop_indexers, errors="ignore")
    # ToDo: proper logging
    except Exception:  # noqa: S110
        pass

    return x


class NodeFailedError(Exception):
    def __init__(self, exception: Exception | str = "NodeFailedError") -> None:
        """The default exception when a node's compute function fails and failsafe mode
        is enable, i.e. the global setting `fail_mode` is not set to `fail`.
        this exception is caught by the foreal processing system and depending on the
        global variable `fail_mode`, leads to process interruption or continuation.

        Args:
            exception (Exception | str, optional): The reason why it failed, e.g. another exception.
                Defaults to None.
        """
        self.exception = exception

    def __str__(self) -> str:
        return str(self.exception)


def descriptor_update(base: Descriptor | dict, update: dict, convert_nestedfrozen: bool = False) -> dict:
    for key, value in update.items():
        if key == "_deklare_attrs" and isinstance(base, Descriptor):
            descriptor_update(base._deklare_attrs, value, convert_nestedfrozen=convert_nestedfrozen)

        if isinstance(base, Descriptor) and key in base:
            field_info = base.__pydantic_fields__[key]
            if DatetimeRange in get_args(field_info.annotation) and isinstance(value, dict):
                value = DatetimeRange(**value)
            elif isinstance(value, dict):
                for t in get_args(field_info.annotation):
                    if issubclass(t, Range):
                        value = Range(**value)
                        break

        if isinstance(value, dict):
            if key not in base:
                base[key] = {}

            if convert_nestedfrozen:
                base[key] = dict(base[key])

            base[key] = descriptor_update(base[key], value, convert_nestedfrozen=convert_nestedfrozen)
        else:
            base[key] = value

    return base


def extract_subgraphs(taskgraph: list[Graph] | Graph, keys: Iterable, match_base_name: bool = False) -> Delayed:
    if not isinstance(taskgraph, list):
        taskgraph = [taskgraph]

    extracted_graph, _ = _extract_graph_and_keys(taskgraph)
    if match_base_name:
        configured_graph_keys = list(extracted_graph.keys())
        new_keys = []
        for k in configured_graph_keys:
            for sk in keys:
                if base_name(sk) == base_name(k):
                    new_keys += [k]
        keys = new_keys

    return Delayed(keys, extracted_graph)


def to_datetime(x: DatetimeScalar, **kwargs: Any) -> NaTType:  # noqa: ANN401
    # overwrites default
    if not kwargs.pop("utc", True):
        warnings.warn("to_datetime overwrites your keyword utc argument and enforces `utc=True`", stacklevel=1)

    return pd.to_datetime(x, utc=True, **kwargs).tz_localize(None)


def is_datetime(x: Any) -> bool:  # noqa: ANN401
    return pd.api.types.is_datetime64_any_dtype(x)


def to_datetime_conditional(x: Any, condition: bool | DatetimeScalar | pd.Timedelta = True, **kwargs: Any) -> xr:  # noqa: ANN401
    # converts x to datetime if condition is true or the object in condition is datetime or timedelta
    if not isinstance(condition, bool):
        condition = is_datetime(condition) or isinstance(condition, pd.Timedelta)

    if condition:
        return to_datetime(x, **kwargs)
    return x


# ToDo: simplify and break up function
def get_segments(  # noqa: C901
    dataset_scope: dict,
    segment_slice: dict,
    segment_stride: dict | None = None,
    reference: dict | None = None,
    mode: str = "overlap",
    minimal_number_of_segments: int = 0,
    timestamps_as_strings: bool = False,
    utc_no_tz: bool = True,
) -> list[dict]:
    # modified from and thanks to xbatcher: https://github.com/rabernat/xbatcher/
    if isinstance(mode, str):
        mode = {dim: mode for dim in segment_slice}

    segment_stride = segment_stride or {}
    reference = reference or {}

    dim_slices = []
    dims = []
    for dim in segment_slice:
        if dim not in dataset_scope:
            continue
        dims += [dim]

        _segment_slice = segment_slice[dim]
        _segment_stride = segment_stride.get(dim, _segment_slice)

        dataset_scope_dim = dataset_scope[dim]
        if not isinstance(dataset_scope_dim, (list, Range, dict)):
            dataset_scope_dim = [dataset_scope_dim]

        if isinstance(dataset_scope_dim, list):
            segment_start = 0
            segment_end = len(dataset_scope_dim)

            if _segment_slice == "full":
                dim_slices += [[dataset_scope_dim]]
                continue

        elif isinstance(dataset_scope[dim], (Range, dict)):
            if isinstance(dataset_scope[dim], Range):
                dataset_scope_dim = dict(dataset_scope[dim])

            if _segment_slice == "full":
                dim_slices += [[dataset_scope_dim]]
                continue

            # make sure _segment_stride and _segment_slice have right orientation
            if not isinstance(_segment_stride, pd.Timedelta):
                if (dataset_scope_dim["end"] - dataset_scope_dim["start"]) * _segment_stride < 0:
                    _segment_stride *= -1
                if _segment_slice * _segment_stride < 0:
                    _segment_slice *= -1

            segment_start = to_datetime_conditional(dataset_scope_dim["start"], _segment_slice)
            segment_end = to_datetime_conditional(dataset_scope_dim["end"], _segment_slice)

            if mode[dim] == "overlap":
                # TODO: add options for closed and open intervals
                # first get the lowest that window that still overlaps with our segment
                segment_start = segment_start - math.floor(_segment_slice / _segment_stride) * _segment_stride
                # then align to the grid if necessary
                if dim in reference:
                    ref_dim = to_datetime_conditional(reference[dim], _segment_slice)
                    segment_start = math.ceil((segment_start - ref_dim) / _segment_stride) * _segment_stride + ref_dim

            elif mode[dim] == "fit":
                if dim in reference:
                    ref_dim = to_datetime_conditional(reference[dim], _segment_slice)
                    segment_start = math.floor((segment_start - ref_dim) / _segment_stride) * _segment_stride + ref_dim
                else:
                    raise RuntimeError(f"mode `fit` requires that dimension {dim} is in reference {reference}")
            else:
                RuntimeError(f"Unknown mode {mode[dim]}. It must be `fit` or `overlap`")

        if isinstance(segment_slice[dim], pd.Timedelta):  # or isinstance(segment_slice[dim], dt.timedelta):
            # TODO: change when xarray #3291 is fixed
            iterator = pd.date_range(segment_start, segment_end, freq=_segment_stride)
            segment_end = pd.to_datetime(segment_end)
        else:
            iterator = range(int(segment_start), int(segment_end), _segment_stride)

        slices = []
        for start in iterator:
            end = start + _segment_stride

            if (
                start <= end
                or (not isinstance(_segment_stride, pd.Timedelta) and _segment_slice < 0 and start >= end)
                or (len(slices) < minimal_number_of_segments and not isinstance(dataset_scope_dim, list))
            ):
                if is_datetime(start):
                    if utc_no_tz:
                        start = pd.to_datetime(start, utc=True).tz_localize(None)
                    if timestamps_as_strings:
                        start = start.isoformat()
                if is_datetime(end):
                    if utc_no_tz:
                        end = pd.to_datetime(end, utc=True).tz_localize(None)
                    if timestamps_as_strings:
                        end = end.isoformat()

                if isinstance(dataset_scope_dim, list):
                    slices.append(dataset_scope_dim[start:end])
                else:
                    slices.append({"start": start, "end": end})
        dim_slices.append(slices)

    all_slices = []
    for slices in itertools.product(*dim_slices):
        selector = {key: dim_slice for key, dim_slice in zip(dims, slices, strict=False)}
        all_slices.append(selector)

    return all_slices
