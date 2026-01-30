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
import warnings
import pandas as pd
from .deskribe import Range
import math
from datetime import datetime

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
    for k, v in indexers.items():
        if not isinstance(v,dict):
            continue
        end_val = v.get("end")
        
        if k not in x.coords or end_val is None:
            continue

        # 1. Grab the underlying pandas index to check metadata
        #    This is instant (does not scan data)
        idx = x.indexes.get(k)
        
        # 2. Case A: Sorted Increasing (Standard Time Series)
        if idx is not None and idx.is_monotonic_increasing:
            if x.sizes[k] > 0:
                # Check last value using .item() for scalar conversion
                # (Faster than numpy array comparison)
                if x[k].isel({k: -1}).item() == end_val:
                    x = x.isel({k: slice(None, -1)})

        # 3. Case B: Sorted Decreasing (Rare, but possible)
        elif idx is not None and idx.is_monotonic_decreasing:
             if x.sizes[k] > 0:
                # In a decreasing list, the "end" value would be at the start (index 0)
                # assuming the user meant "exclude values <= end"
                # If the user meant "exclude values >= end", logic flips. 
                # Assuming standard "drop this specific label" logic:
                if x[k].isel({k: 0}).item() == end_val:
                    x = x.isel({k: slice(1, None)})

        # 4. Case C: Unsorted / Complex (The Fallback)
        else:
            # Fallback to the masking method (faster than drop_sel)
            mask = x[k] != end_val
            if not mask.all():
                x = x.isel({k: mask})

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


def to_datetime(x, **kwargs):
    # overwrites default
    utc = kwargs.pop("utc", True)
    if not utc:
        warnings.warn(
            "to_datetime overwrites your keyword utc argument and enforces `utc=True`"
        )
    return pd.to_datetime(x, utc=True, **kwargs).tz_localize(None)


def is_datetime(x):
    """Checks if input is a datetime-like scalar or array."""
    if isinstance(x, (pd.Timestamp, datetime)): # check also for , np.datetime64
        return True
    return pd.api.types.is_datetime64_any_dtype(x)

def to_datetime_conditional(x, condition=True, **kwargs):
    # converts x to datetime if condition is true or the object in condition is datetime or timedelta
    if not isinstance(condition, bool):
        condition = is_datetime(condition) or isinstance(condition, pd.Timedelta)

    if condition:
        return to_datetime(x, **kwargs)
    return x


def get_segments(
    dataset_scope,
    segment_slice,
    segment_stride=None,
    reference=None,
    mode="overlap",
    minimal_number_of_segments=0,
    timestamps_as_strings=False,
    utc_no_tz=True,
):
    # modified from and thanks to xbatcher: https://github.com/rabernat/xbatcher/
    if isinstance(mode, str):
        mode = {dim: mode for dim in segment_slice}

    if segment_stride is None:
        segment_stride = {}

    if reference is None:
        reference = {}

    dim_slices = []
    dims = []
    for dim in segment_slice:
        if dim not in dataset_scope:
            continue
        dims += [dim]

        _segment_slice = segment_slice[dim]
        _segment_stride = segment_stride.get(dim, _segment_slice)
#        print(_segment_slice,_segment_stride)
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
                if (
                    dataset_scope_dim["end"] - dataset_scope_dim["start"]
                ) * _segment_stride < 0:
                    _segment_stride *= -1
                if _segment_slice * _segment_stride < 0:
                    _segment_slice *= -1

            segment_start = to_datetime_conditional(
                dataset_scope_dim["start"], _segment_slice
            )
            segment_end = to_datetime_conditional(
                dataset_scope_dim["end"], _segment_slice
            )

            if mode[dim] == "overlap":
                # 1. Determine the "epsilon" (smallest unit) for the current data type
                if is_datetime(segment_start):
                    epsilon = pd.Timedelta(nanoseconds=1)
                else:
                    epsilon = 1 if isinstance(segment_start, int) else 1e-9

                # 2. Handle reference alignment only if it exists
                ref_val = 0
                if dim in reference:
                    ref_val = to_datetime_conditional(reference[dim], _segment_slice)
                
                # 3. Calculate how many strides to back up.
                # We subtract epsilon to ensure that if segment_start is EXACTLY on 
                # the boundary of the next window, we don't count it as an overlap.
                num_strides = math.floor((segment_start - ref_val - epsilon) / _segment_stride)
                segment_start = ref_val + (num_strides * _segment_stride)

                # 4. Optional: Grid alignment (only if reference is provided)
                if dim in reference:
                    segment_start = (
                        math.ceil((segment_start - ref_val) / _segment_stride)
                        * _segment_stride
                        + ref_val
                    )
            elif mode[dim] == "fit":
                if dim in reference:
                    ref_dim = to_datetime_conditional(reference[dim], _segment_slice)
                    segment_start = (
                        math.floor((segment_start - ref_dim) / _segment_stride)
                        * _segment_stride
                        + ref_dim
                    )
                else:
                    raise RuntimeError(
                        f"mode `fit` requires that dimension {dim} is in reference {reference}"
                    )
            else:
                RuntimeError(f"Unknown mode {mode[dim]}. It must be `fit` or `overlap`")

        if isinstance(segment_slice[dim], pd.Timedelta):
            # Determine the smallest possible step to make the end exclusive
            epsilon = pd.Timedelta(nanoseconds=1)
            
            # We stop at (segment_end - epsilon) to ensure segment_end is never 
            # included as a 'start' point.
            iterator = pd.date_range(
                start=segment_start, 
                end=segment_end - epsilon, 
                freq=_segment_stride
            )
            segment_end = pd.to_datetime(segment_end)
        else:
            # Python's range(start, stop) is already exclusive of 'stop'
            iterator = range(int(segment_start), int(segment_end), int(_segment_stride))
            
        slices = []
        for start in iterator:
            end = start + _segment_slice

            if (
                start <= end
                or (
                    not isinstance(_segment_stride, pd.Timedelta)
                    and _segment_slice < 0
                    and start >= end
                )
                or (
                    len(slices) < minimal_number_of_segments
                    and not isinstance(dataset_scope_dim, list)
                )
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

    import itertools

    all_slices = []
    for slices in itertools.product(*dim_slices):
        selector = {key: slice for key, slice in zip(dims, slices)}
        all_slices.append(selector)

    return all_slices
