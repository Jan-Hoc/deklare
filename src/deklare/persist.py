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

import io
import json
import warnings
from copy import copy, deepcopy
from pathlib import Path
from threading import Lock
from typing import Callable, TypeVar, Generic
from abc import ABC, abstractmethod

import math
import pandas as pd
import zarr
from cachetools import LRUCache
from compress_pickle import dump, load

# TODO: can we implement our own hash function for deskriptors to reduce dependency on dask?
from dask.base import tokenize

from .core import task
from .deskribe import Range
from .utils import (
    NodeFailedException,
    dict_update,
    exclusive_indexing,
    indexers_to_slices,
)

# TODO: Redesign to StorageManager
T = TypeVar('T')
class StorageManager(ABC, Generic[T]):
    @abstractmethod
    def write(self, data: T) -> io.BytesIO:
        """Takes the data and returns it as a buffer containing data in desired format

        Args:
            data (T): data to be saved

        Returns:
            io.BytesIO: buffer containing data in desired format
        """
        pass

    @abstractmethod
    def read(self, buffer: io.BytesIO) -> T:
        """Read the data in buffer and returns the read data in the same format as it was received in write

        Args:
            buffer (io.BytesIO): buffer containing saved data

        Returns:
            T: data in buffer
        """
        pass

class PickleStorageManager(StorageManager):
    def __init__(self, compression="gzip"):
        super().__init__()
        self.compression = compression

    def write(self, data: T) -> io.BytesIO:
        buffer = io.BytesIO()
        dump(object, buffer, compression=self.compression)
        return buffer

    def read(self, buffer: io.BytesIO) -> T:
        return load(buffer, compression=self.compression)


@task()
class Persister():
    def __init__(
        self,
        store=None,
        storage_manager: None|StorageManager=None,
        selected_keys=None,
        force_update=False,
        use_memorycache=True,
    ):
        super().__init__(force_update=force_update, use_memorycache=use_memorycache)
        if isinstance(store, str) or isinstance(store, Path):
            store = zarr.DirectoryStore(store)
        self.store = store
        self.storage_manager = storage_manager
        self.cache = LRUCache(10)
        if selected_keys is None:
            # use all keys as hash
            pass
        self._mutex = Lock()

    def configure(self, request: T|None=None):
        request_hash = self.get_hash(request)

        # compute action defaults to passthrough
        request["self"]["action"] = "passthrough"

        if request["self"].get("bypass", False):
            # set to passthrough -> nothing will happen
            return request

        # propagate the request_hash to the compute function
        request["self"]["request_hash"] = request_hash

        # reload and rewrite the chunk if requested
        if request["self"].get("force_update", False):
            request["self"]["action"] = "store"
            return request

        with self._mutex:
            if (
                request["self"].get("use_memorycache", True)
                and request_hash in self.cache
            ):
                request["remove_dependencies"] = True
                # set the compute action to load
                request["self"]["action"] = "load_from_cache"
                return request

            # while holding the mutex, we need to check if the file exists
            if request_hash in self.store:
                # remove previous node since we are going to load from disk
                request["remove_dependencies"] = True

                # set the compute action to load
                request["self"]["action"] = "load"
                return request
            elif "fail/" + request_hash in self.store:
                # remove previous node since we are going to load the fail info from disk
                request["remove_dependencies"] = True
                request["self"]["request_hash"] = "fail/" + request_hash

                # set the compute action to load
                request["self"]["action"] = "load"
                return request

            # TODO: check if the file will be written to already?

            request["self"]["action"] = "store"

        return request

    def compute(self, data: T|None=None, **request):
        if request["action"] == "load_from_cache":
            with self._mutex:
                cached = self.cache[request["request_hash"]]
            return cached
        elif request["action"] == "load":
            buffer = io.BytesIO(self.store[request["request_hash"]])
            data = self.storage_manager.read(buffer)

            with self._mutex:
                self.cache[request["request_hash"]] = data

            return data
        elif request["action"] == "store":
            try:
                self.cache[request["request_hash"]] = data

                # write to file
                if isinstance(data, NodeFailedException):
                    buffer = self.storage_manager.write(data)
                    self.store["fail/" + request["request_hash"]] = buffer.getvalue()
                else:
                    if isinstance(data, str):
                        raise RuntimeError(f"something wrong {data}")
                    buffer = self.storage_manager.write(data)
                    self.store[request["request_hash"]] = buffer.getvalue()

            except Exception as e:
                print("Error during hashpersister", repr(e))

            with self._mutex:
                self.cache[request["request_hash"]] = data

            return data
        elif request["action"] == "passthrough":
            return data
        else:
            raise NodeFailedException("A bug in HashPersister. Please report.")

    def is_valid(self, request: dict):
        """Checks if persisted object for `request`
        exists and is valid (i.e. is not of type NodeFailedException).

        Args:
            request (dict): The request that should be checked

        Returns:
            boolean or None: Returns false if the persisted item is of type NodeFailedException
                             Returns None if the request has not been persisted yet.
        """
        request_hash = self.get_hash(request)

        if "fail/" + request_hash in self.store:
            return False

        if request_hash in self.store:
            return True

        return None

    def get_hash(self, request: dict) -> str:
        """returns the hash of the request

        Args:
            request (dict): request

        Returns:
            str: hash of the requenst
        """
        r = {k: v for k, v in request.items() if k != "self"}
        s = json.dumps(r, sort_keys=True, skipkeys=True, default=self._string_timestamp)
        request_hash = tokenize(s)

        return request_hash

    def _string_timestamp(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        else:
            return str(o)

def to_datetime(x, **kwargs):
    # overwrites default
    utc = kwargs.pop("utc", True)
    if not utc:
        warnings.warn(
            "to_datetime overwrites your keyword utc argument and enforces `utc=True`"
        )
    return pd.to_datetime(x, utc=True, **kwargs).tz_localize(None)


def is_datetime(x):
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

        dataset_scope_dim = dataset_scope[dim]
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

            segment_start = to_datetime_conditional(
                dataset_scope_dim["start"], segment_slice[dim]
            )
            segment_end = to_datetime_conditional(
                dataset_scope_dim["end"], segment_slice[dim]
            )

            if mode[dim] == "overlap":
                # TODO: add options for closed and open intervals
                # first get the lowest that window that still overlaps with our segment
                segment_start = (
                    segment_start
                    - math.floor(_segment_slice / _segment_stride) * _segment_stride
                )
                # then align to the grid if necessary
                if dim in reference:
                    ref_dim = to_datetime_conditional(
                        reference[dim], segment_slice[dim]
                    )
                    segment_start = (
                        math.ceil((segment_start - ref_dim) / _segment_stride)
                        * _segment_stride
                        + ref_dim
                    )
                segment_end = segment_end + _segment_slice
            elif mode[dim] == "fit":
                if dim in reference:
                    ref_dim = to_datetime_conditional(
                        reference[dim], segment_slice[dim]
                    )
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

        if isinstance(
            segment_slice[dim], pd.Timedelta
        ):  # or isinstance(segment_slice[dim], dt.timedelta):
            # TODO: change when xarray #3291 is fixed
            iterator = pd.date_range(segment_start, segment_end, freq=_segment_stride)
            segment_end = pd.to_datetime(segment_end)
        else:
            iterator = range(segment_start, segment_end, _segment_stride)

        slices = []
        for start in iterator:
            end = start + _segment_slice
            if end <= segment_end or (
                len(slices) < minimal_number_of_segments
                and not isinstance(dataset_scope_dim, list)
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


try:
    import xarray as xr
except ImportError:
    warnings.warn("Install xarray to use the default merge function of ChunkPersister")


def merge_xarray(data, request):
    data = [d for d in data if d is not None]
    for i in range(len(data)):
        if hasattr(data[i], "name") and (not data[i].name or data[i].name is None):
            data[i].name = "data"
    merged_dataset = xr.merge(data)
    # if hasattr(data[0], "name"):
    #     merged_dataset = merged_dataset[data[0].name]
    indexers = {}
    for coord in merged_dataset.coords:
        if coord in request:
            indexers[coord] = request[coord]
    slices = indexers_to_slices(indexers)
    section = merged_dataset.sel(slices)
    section = exclusive_indexing(section, indexers)
    return section


@task()
class ChunkPersister(Persister):
    def __init__(
        self,
        store,
        dim: str = "time",
        # classification_scope:dict | Callable[...,dict]=None,
        segment_slice: dict | Callable[..., dict] = None,
        # segment_stride:dict|Callable[...,dict]=None,
        dataset_scope: dict | Callable[..., dict] = None,
        mode: str = "overlap",
        reference: dict = None,
        force_update=False,
        merge_function=None,
        storage_manager: StorageManager=PickleStorageManager,
    ):
        """Chunks every incoming dekriptor into subchunks if deskriptor is larger than segment_slice
         or extends the deskriptor to the respective chunksize if deskriptor is smaller than segment_slice

        Args:
            store (_type_): _description_
            dim (str, optional): _description_. Defaults to "time".
            segment_slice (dict | Callable[..., dict], optional): A dictionary containing an entry for each dimension that should be chunked. Each entry is the respective chunk size given in the units of the expected dimension of the deskriptor. For example, for a time dimension you can use pd.Timedelta. Defaults to None.
            dataset_scope (dict | Callable[...,dict], optional): The extend of the chunking. If None, the incoming deskriptor will be used as the scope. If only selected dimensions are given as dataset_scope, the scope for the other dimensions will be choosen from the incoming deskriptor. Defaults to None.
            mode (str, optional): _description_. Defaults to "overlap".
            reference (dict, optional): _description_. Defaults to None.
            force_update (bool, optional): _description_. Defaults to False.
            merge_function (_type_, optional): _description_. Defaults to None.
        """
        # if callable(classification_scope):
        #     self.classification_scope = classification_scope
        #     classification_scope = None
        # else:
        #     self.classification_scope = None

        if callable(segment_slice):
            self.segment_slice = segment_slice
            segment_slice = None
        else:
            self.segment_slice = None

        # if callable(segment_stride):
        #     self.segment_stride = segment_stride
        #     segment_stride = None
        # else:
        #     self.segment_stride = None

        self.merge = merge_function
        if self.merge is None:
            self.merge = merge_xarray

        super().__init__(
            dim=dim,
            # classification_scope=classification_scope,
            segment_slice=segment_slice,
            # segment_stride=segment_stride,
            dataset_scope=dataset_scope,
            mode=mode,
            reference=reference,
            force_update=force_update,
        )

        if isinstance(store, str) or isinstance(store, Path):
            store = zarr.DirectoryStore(store)
        self.store = store

        self.storage_manager = storage_manager

    def __dask_tokenize__(self):
        return (ChunkPersister,)

    def configure(self, request=None):
        rs = request["self"]

        def get_value(attr_name):
            # decide if we use the attribute provided in the request or
            # from a callback provided at initialization

            if rs.get(attr_name, None) is None:
                # there is no attribute in the request, check for callback
                callback = getattr(self, attr_name)
                if callback is not None and callable(callback):
                    value = callback(request)
                else:
                    RuntimeError("No valid classification_scope provided")
            else:
                value = rs[attr_name]
            return value

        dataset_scope = copy(rs)
        if rs.get("dataset_scope", None) is not None:
            dataset_scope.update(rs["dataset_scope"])
        segment_slice = get_value("segment_slice")
        # segment_stride = get_value("segment_stride")
        segments = get_segments(
            dataset_scope,
            segment_slice,
            # segment_stride,
            reference=rs["reference"],
            mode=rs["mode"],
            timestamps_as_strings=True,
            minimal_number_of_segments=1,
        )
        cloned_requests = []
        cloned_persisters = []
        for segment in segments:
            segment_request = deepcopy(request)
            if "self" in segment_request:
                del segment_request["self"]
            dict_update(segment_request, segment)
            cloned_requests += [segment_request]
            cloned_persister = Persister(
                store=self.store,
                storage_manager=self.storage_manager
            )
            cloned_persister.dask_key_name = self.dask_key_name + "_persister"
            dict_update(
                segment_request,
                {
                    "config": {
                        "keys": {
                            self.dask_key_name + "_persister": {
                                "force_update": rs.get("force_update", False)
                            }
                        }
                    }
                },
            )
            cloned_persisters += [cloned_persister.compute]

        # Insert predecessor
        # new_request = {}
        request["clone_dependencies"] = cloned_requests
        request["insert_predecessor"] = cloned_persisters

        return request

    def compute(self, *data, **request):
        def unpack_list(inputlist):
            new_list = []
            for item in inputlist:
                if isinstance(item, list):
                    new_list += unpack_list(item)
                else:
                    new_list += [item]
            return new_list

        data = unpack_list(data)
        success = [d for d in data if not isinstance(d, NodeFailedException)]

        if not success:
            failed = [str(d) for d in data if isinstance(d, NodeFailedException)]
            raise RuntimeError(f"Failed to load data. Reason: {failed}")
        section = self.merge(success, request)
        return section
