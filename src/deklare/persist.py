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
import os
import json
import warnings
from copy import copy, deepcopy
from pathlib import Path
from threading import Lock
from typing import Callable, TypeVar, Generic
from abc import ABC, abstractmethod

import math
import pandas as pd
from cachetools import LRUCache
from compress_pickle import dump, load

# TODO: can we implement our own hash function for deskriptors to reduce dependency on dask?
from dask.base import tokenize

import pystac
from shapely.geometry import Polygon, mapping

from .core import task
from .utils import (
    NodeFailedException,
    dict_update,
    exclusive_indexing,
    indexers_to_slices,
)

import fsspec

from .utils import get_segments


class StacIO(pystac.StacIO):
    def __init__(self, store):
        self.store = store

    def read_text(self, source: pystac.utils.HREF, *args, **kwargs) -> str:
        """Reads the data at `source` stored in the store

        Args:
            source : The source to read from.

        Returns:
            str: The text contained in the file at the location specified by the uri.
        """
        str_src = str(source)
        return self.store[str_src].decode()

    def write_text(self, dest: pystac.utils.HREF, txt: str, *args, **kwargs) -> None:
        """writes the data of `txt` into the store to `dest`

        Args:
            dest : The destination to write to.
            txt : The text to write.
        """
        str_dest = str(dest)
        self.store[str_dest] = txt.encode()


T = TypeVar("T")


class StorageManager(ABC, Generic[T]):
    # @abstractmethod
    # def write_buffer(self, data: T, buffer: io.BytesIO =None) -> io.BytesIO:
    #     """Takes the data and returns it as a buffer containing data in desired format

    #     Args:
    #         data (T): data to be saved

    #     Returns:
    #         io.BytesIO: buffer containing data in desired format
    #     """
    #     pass

    # @abstractmethod
    # def read_buffer(self, buffer: io.BytesIO) -> T:
    #     """Read the data in buffer and returns the read data in the same format as it was received in write

    #     Args:
    #         buffer (io.BytesIO): buffer containing saved data

    #     Returns:
    #         T: data in buffer
    #     """
    #     pass

    @abstractmethod
    def write(self, file, data: T):
        """Takes the data and writes it to the file

        Args:
            data (T): data to be saved

        """
        pass

    @abstractmethod
    def read(self, file) -> T:
        """Returns the read data in the same format as it was received in write

        Args:
            file (): file containing saved data

        Returns:
            T: data from file
        """
        pass

    @abstractmethod
    def file_info(self) -> tuple[str, str]:
        """return media type and text describtion of file saved in `write` function
        media type should be registered in https://www.iana.org/assignments/media-types/media-types.xhtml e.g. `image/tiff`
        description should be human readable with information needed to read file

        Returns:
            tuple[str, str]: (media type, description)
        """
        pass


class PickleStorageManager(StorageManager):
    def __init__(self, compression="gzip"):
        super().__init__()
        self.compression = compression

    # def write_buffer(self, data: T, buffer: io.BytesIO =None) -> io.BytesIO:
    #     if buffer is None:
    #         buffer = io.BytesIO()
    #     dump(data, buffer, compression=self.compression)
    #     return buffer

    # def read_buffer(self, buffer: io.BytesIO) -> T:
    #     return load(buffer, compression=self.compression)

    def write(self, file, data: T) -> io.BytesIO:
        dump(data, file, compression=self.compression)

    def read(self, file) -> T:
        return load(file, compression=self.compression)

    def file_info(self) -> tuple[str, str]:
        return (
            "application/octet-stream",
            f"Pickle file using compression {self.compression}",
        )


@task()
class Persister:
    def __init__(
        self,
        store=None,
        storage_manager: None | StorageManager = None,
        stac_io: StacIO = None,
        selected_keys=None,
        force_update=False,
        use_memorycache=True,
        cache=None,
        global_lock=None,
        save_metadata=False,
    ):
        super().__init__(force_update=force_update, use_memorycache=use_memorycache)
        if isinstance(store, str) or isinstance(store, Path):
            store = fsspec.get_mapper(store)
        self.store = store
        self.storage_manager = storage_manager

        if cache is None:
            cache = LRUCache(10)
        self.cache = cache

        if selected_keys is None:
            # use all keys as hash
            pass

        self.stac_io = stac_io
        self._global_lock = global_lock
        self._mutex = Lock()
        self.save_metadata = save_metadata

    def configure(self, deskriptor: T | None = None):
        deskriptor_hash = self.get_hash(deskriptor)
        data_path = f"data/{deskriptor_hash}"

        # compute action defaults to passthrough
        deskriptor["self"]["action"] = "passthrough"

        if deskriptor["self"].get("bypass", False):
            # set to passthrough -> nothing will happen
            return deskriptor

        # propagate the deskriptor_hash to the compute function
        deskriptor["self"]["deskriptor_hash"] = deskriptor_hash

        # reload and rewrite the chunk if deskriptored
        if deskriptor["self"].get("force_update", False):
            deskriptor["self"]["action"] = "store"
            return deskriptor

        with self._mutex:
            if (
                deskriptor["self"].get("use_memorycache", True)
                and data_path in self.cache
            ):
                deskriptor["remove_dependencies"] = True
                # set the compute action to load
                deskriptor["self"]["action"] = "load_from_cache"
                return deskriptor

            if self.store is None:
                return deskriptor
            
            # while holding the mutex, we need to check if the file exists
            if data_path in self.store:
                # remove previous node since we are going to load from disk
                deskriptor["remove_dependencies"] = True

                # set the compute action to load
                deskriptor["self"]["action"] = "load"
                return deskriptor
            elif "fail/" + deskriptor_hash in self.store:
                # remove previous node since we are going to load the fail info from disk
                deskriptor["remove_dependencies"] = True
                deskriptor["self"]["deskriptor_hash"] = "fail/" + deskriptor_hash

                # set the compute action to load
                deskriptor["self"]["action"] = "load"
                return deskriptor

            # TODO: check if the file will be written to already?

            deskriptor["self"]["action"] = "store"

        return deskriptor

    def compute(self, data: T | None = None, **deskriptor):
        if deskriptor["action"] == "passthrough":
            return data
    
        if self.store is not None:
            self.store.dirfs.mkdirs("data/", exist_ok=True)
            data_path = f"data/{deskriptor['deskriptor_hash']}"

        if deskriptor["action"] == "load_from_cache":
            with self._mutex:
                cached = self.cache[data_path]
            return cached
        elif deskriptor["action"] == "load":
            f = self.store.dirfs.open(data_path)
            data = self.storage_manager.read(f)

            with self._mutex:
                self.cache[data_path] = data

            return data
        elif deskriptor["action"] == "store":
            with self._mutex:
                self.cache[data_path] = data

            if self.store is None:
                return data 
            
            try:
                # in this case we assume that the second element is additional metadata for the STAC item
                if (
                    isinstance(data, tuple)
                    and len(data) == 2
                    and isinstance(data[1], dict)
                ):
                    item_metadata = data[1]
                    data = data[0]
                else:
                    item_metadata = {}

                # write to file
                if isinstance(data, NodeFailedException):
                    # buffer = self.storage_manager.write_buffer(data)
                    # self.store["fail/" + deskriptor["deskriptor_hash"]] = buffer.getvalue()
                    self.store.dirfs.mkdirs("fail/", exist_ok=True)
                    with self.store.dirfs.open(
                        "fail/" + deskriptor["deskriptor_hash"], "wb"
                    ) as f:
                        self.storage_manager.write(f, data)

                else:
                    if isinstance(data, str):
                        raise RuntimeError(f"something wrong {data}")

                    # buffer = self.storage_manager.write_buffer(data)
                    # self.store[data_path] = buffer.getvalue()
                    try:
                        with self.store.dirfs.open(data_path, "wb") as f:
                            self.storage_manager.write(f, data)
                    except Exception as e:
                        self.store.dirfs.rm(data_path)
                        raise e

                    if self.save_metadata:
                        self._save_metadata(deskriptor, item_metadata)

            except Exception as e:
                print("Error during Persister", repr(e))


            return data
        else:
            raise NodeFailedException("A bug in Persister. Please report.")

    def is_valid(self, deskriptor: dict):
        """Checks if persisted object for `deskriptor`
        exists and is valid (i.e. is not of type NodeFailedException).

        Args:
            deskriptor (dict): The deskriptor that should be checked

        Returns:
            boolean or None: Returns false if the persisted item is of type NodeFailedException
                             Returns None if the deskriptor has not been persisted yet.
        """
        deskriptor_hash = self.get_hash(deskriptor)

        if "fail/" + deskriptor_hash in self.store:
            return False

        if deskriptor_hash in self.store:
            return True

        return None

    def get_hash(self, deskriptor: dict) -> str:
        """returns the hash of the deskriptor

        Args:
            deskriptor (dict): deskriptor

        Returns:
            str: hash of the requenst
        """
        r = {k: v for k, v in deskriptor.items() if k != "self"}
        s = json.dumps(
            r, sort_keys=True, skipkeys=True, default=Persister._string_timestamp
        )
        deskriptor_hash = tokenize(s)

        return deskriptor_hash

    def _save_metadata(self, deskriptor: dict, item_metadata: dict) -> None:
        """saves metadata for given chunk using STAC (https://stacspec.org/)

        Args:
            deskriptor (dict): the deskriptor containing the temporal and spacial boundaries
            item_metadata(dict): additional metadata passed by loader to save in STAC item
        """

        kwargs = Persister._gen_item_kwargs(deskriptor, item_metadata)

        item = pystac.Item(**kwargs)

        file_info = self.storage_manager.file_info()

        asset = pystac.Asset(
            href=f"./../../data/{deskriptor['deskriptor_hash']}",
            description=file_info[1],
            media_type=file_info[0],
            roles=["data"],
        )
        item.add_asset(key="data", asset=asset)

        # save in collection and avoid concurrency issues
        with self._global_lock:
            collection = pystac.Collection.from_file(
                "stac/collection.json", self.stac_io
            )  # pretending location is absolute to stop stac from changing path
            collection.add_item(item)
            collection.save(
                catalog_type=pystac.CatalogType.SELF_CONTAINED,
                dest_href="stac",  # pretend path is absolute so pystac doesnt try and change it
                stac_io=self.stac_io,
            )

    def _gen_description(deskriptor) -> str:
        """generate human readable description for STAC Item of chunk

        Returns:
            str: STAC Item description
        """
        start_time = deskriptor["time"]["start"].isoformat()
        end_time = deskriptor["time"]["end"].isoformat()
        variable_string = ", ".join(deskriptor["variable"])
        latitude_string = f"{deskriptor['latitude']['start']} to {deskriptor['latitude']['end']} latitude"
        longitude_string = f"{deskriptor['longitude']['start']} to {deskriptor['longitude']['end']} longitude"

        description = (
            f"This chunk contains data for the variable(s) {variable_string}, "
            f"collected from {start_time} to {end_time}, "
            f"covering the geographic region defined by {latitude_string} and {longitude_string}"
        )

        return description

    def _gen_item_kwargs(deskriptor: dict, item_metadata: dict) -> dict:
        id = deskriptor["deskriptor_hash"]
        bbox = [
            deskriptor["longitude"]["start"],
            deskriptor["latitude"]["end"],
            deskriptor["longitude"]["end"],
            deskriptor["latitude"]["start"],
        ]
        footprint = mapping(
            Polygon(
                [
                    [bbox[0], bbox[1]],  # lower left corner
                    [bbox[0], bbox[3]],  # upper left corner
                    [bbox[2], bbox[3]],  # upper right corner
                    [bbox[2], bbox[1]],  # lower right corner
                    [bbox[0], bbox[1]],  # lower left corner
                ]
            )
        )
        start_time = deskriptor["time"]["start"].to_pydatetime()
        end_time = deskriptor["time"]["end"].to_pydatetime()
        variables = deskriptor["variable"]

        kwargs = {
            "id": id,
            "geometry": footprint,
            "bbox": bbox,
            "datetime": None,
            "start_datetime": start_time,
            "end_datetime": end_time,
            "properties": {
                "description": Persister._gen_description(deskriptor),
                "variables": variables,
            },
        }

        blocked_keys = kwargs.keys()
        for k, v in item_metadata.items():
            if k not in blocked_keys:
                kwargs[k] = v
            elif k == "properties" and isinstance(v, dict):
                for k_p, v_p in v.items():
                    if k_p != "variables":
                        kwargs[k][k_p] = v_p

        return kwargs

    def _string_timestamp(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        else:
            return str(o)


try:
    import xarray as xr
except ImportError:
    warnings.warn("Install xarray to use the default merge function of ChunkPersister")


def merge_xarray(data, deskriptor):
    data = [d for d in data if d is not None]
    if len(data) == 1:
        merged_dataset = data[0]
    else:
        for i in range(len(data)):
            if hasattr(data[i], "name") and (not data[i].name or data[i].name is None):
                data[i].name = "data"
# merged_dataset = xr.merge(data)
        merged_dataset = xr.concat(data, dim="time")

    # if hasattr(data[0], "name"):
    #     merged_dataset = merged_dataset[data[0].name]
    indexers = {}
    for coord in merged_dataset.indexes:
        if coord in deskriptor:
            indexers[coord] = deskriptor[coord]
    slices = indexers_to_slices(indexers)
    section = merged_dataset.sel(slices)
    section = exclusive_indexing(section, indexers)
    return section


@task()
class ChunkPersister:
    def __init__(
        self,
        store=None,
        filesystem=None,
        dim: str = "time",
        # classification_scope:dict | Callable[...,dict]=None,
        segment_slice: dict | Callable[..., dict] = None,
        segment_stride: dict | Callable[..., dict] = None,
        dataset_scope: dict | Callable[..., dict] = None,
        mode: str = "overlap",
        reference: dict = None,
        force_update=False,
        merge_function=None,
        storage_manager: StorageManager = PickleStorageManager(),
        collection_metadata: dict = {},
        save_metadata=False,
        use_memorycache=True,
        cache = None
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
            storage_manager (StorageManager, optional): Instance handling reading and writing of data. Defaults to PickleStorageManager.
            collection_metadata (dict, optional): Further kwargs for STAC collection. May contain keys ['id', 'title', 'keywords', 'license', 'links', 'providers']. For 'links' and 'providers' lists of either corresponding STAC objects or dicts that can be used as kwargs to construct them. Defaults to {}.
        """
        # if callable(classification_scope):
        #     self.classification_scope = classification_scope
        #     classification_scope = None
        # else:
        #     self.classification_scope = None

        self.use_memorycache = use_memorycache
        if cache is None:
            cache = LRUCache(10)
        self.cache = cache

        if callable(segment_slice):
            self.segment_slice = segment_slice
            segment_slice = None
        else:
            self.segment_slice = None

        if callable(segment_stride):
            self.segment_stride = segment_stride
            segment_stride = None
        else:
            self.segment_stride = None

        self.merge = merge_function
        if self.merge is None:
            self.merge = merge_xarray

        self.save_metadata = save_metadata

        super().__init__(
            dim=dim,
            # classification_scope=classification_scope,
            segment_slice=segment_slice,
            segment_stride=segment_stride,
            dataset_scope=dataset_scope,
            mode=mode,
            reference=reference,
            force_update=force_update,
        )

        if filesystem is None and store is None:
            raise RuntimeError("Either filesystem or store must be provided")

        self.filesystem = filesystem

        if isinstance(store, str) or isinstance(store, Path):
            store = fsspec.get_mapper(store)

        if store is None:
            store = self.filesystem.get_mapper()

        self.store = store

        self.stac_io = StacIO(store=store)

        if self.save_metadata:
            # create collection if doesn't exist
            try:
                collection = pystac.Collection.from_file(
                    "stac/collection.json", self.stac_io
                )
            except:
                collection_metadata = ChunkPersister._process_collection_metadata(
                    collection_metadata
                )
                collection = pystac.Collection(**collection_metadata[0])

                for l in collection_metadata[1]:
                    collection.add_link(l)

                collection.normalize_and_save(
                    root_href="stac",  # pretend path is absolute so pystac doesnt try and change it
                    catalog_type=pystac.CatalogType.SELF_CONTAINED,
                    stac_io=self.stac_io,
                )

        self.storage_manager = storage_manager
        self.mutex = Lock()

    def __dask_tokenize__(self):
        return (ChunkPersister,)

    def configure(self, deskriptor=None):
        rs = deskriptor["self"]
        if rs.get("bypass",False):
            return deskriptor 
        
        def get_value(attr_name):
            # decide if we use the attribute provided in the deskriptor or
            # from a callback provided at initialization
            value = None
            if rs.get(attr_name, None) is None:
                # there is no attribute in the deskriptor, check for callback
                callback = getattr(self, attr_name)
                if callback is not None and callable(callback):
                    value = callback(deskriptor)
                else:
                    # not passing segment_stride is okay
                    if attr_name == "segment_stride":
                        return None
                    raise RuntimeError("No valid {attr_name} provided")
            else:
                value = rs[attr_name]
            return value

        dataset_scope = copy(rs)
        if rs.get("dataset_scope", None) is not None:
            dataset_scope.update(rs["dataset_scope"])
        segment_slice = get_value("segment_slice")
        segment_stride = get_value("segment_stride")
        segments = get_segments(
            dataset_scope,
            segment_slice,
            segment_stride,
            reference=rs["reference"],
            mode=rs["mode"],
            timestamps_as_strings=True,
            minimal_number_of_segments=1,
        )
        cloned_deskriptors = []
        cloned_persisters = []
        for segment in segments:
            segment_deskriptor = deepcopy(deskriptor)
            if "self" in segment_deskriptor:
                del segment_deskriptor["self"]
            dict_update(segment_deskriptor, segment)
            cloned_deskriptors += [segment_deskriptor]
            cloned_persister = Persister(
                store=self.store,
                storage_manager=self.storage_manager,
                stac_io=self.stac_io,
                global_lock=self.mutex,
                save_metadata=self.save_metadata,
                cache = self.cache,
                use_memorycache=self.use_memorycache
            )
            cloned_persister.dask_key_name = self.dask_key_name + "_persister"
            dict_update(
                segment_deskriptor,
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
        # new_deskriptor = {}
        deskriptor["clone_dependencies"] = cloned_deskriptors
        deskriptor["insert_predecessor"] = cloned_persisters

        return deskriptor

    def compute(self, *data, **deskriptor):
        def unpack_list(inputlist):
            new_list = []
            for item in inputlist:
                if isinstance(item, (tuple, list)):
                    new_list += unpack_list(item)
                else:
                    new_list += [item]
            return new_list

        data = unpack_list(data)
        success = [d for d in data if not isinstance(d, NodeFailedException)]

        if not success:
            failed = [str(d) for d in data if isinstance(d, NodeFailedException)]
            raise RuntimeError(f"Failed to load data. Reason: {failed}")

        if self.save_metadata:
            # update extents
            with self.mutex:
                collection = pystac.Collection.from_file(
                    "stac/collection.json", self.stac_io
                )  # pretend path is absolute so pystac doesnt try and change it
                collection.update_extent_from_items()
                collection.save(
                    catalog_type=pystac.CatalogType.SELF_CONTAINED,
                    dest_href="stac",  # pretend path is absolute so pystac doesnt try and change it
                    stac_io=self.stac_io,
                )

        section = self.merge(success, deskriptor)
        return section

    def _process_collection_metadata(
        collection_metadata: dict = {},
    ) -> tuple[dict, list]:
        """return dict with arguments to create pystac.Collection

        Args:
            collection_metadata (dict, optional): Dict containing keyword arguments for pystac.Collection constructor. Defaults to {}.

        Returns:
            (dict, list): dict with list of keyword arguments, list of pystac.Link objects to add to collection
        """

        links = []
        if "links" in collection_metadata and isinstance(
            collection_metadata["links"], list
        ):
            for l in collection_metadata["links"]:
                if isinstance(l, dict):
                    l = pystac.Link(**l)
                elif not isinstance(l, pystac.Link):
                    continue
                links.append(l)

        kwargs = {
            "id": "",
            "description": "",
            "extent": pystac.Extent(
                spatial=pystac.SpatialExtent([None]),
                temporal=pystac.TemporalExtent([[None, None]]),
            ),
            "title": "",
            "catalog_type": pystac.CatalogType.SELF_CONTAINED,
            "license": "",
            "keywords": None,
            "providers": None,
        }

        blocked_kwargs = ["extent", "catalog_type", "links"]
        for k in collection_metadata:
            if k not in blocked_kwargs:
                kwargs[k] = collection_metadata[k]

        if isinstance(kwargs["providers"], list):
            providers = []
            for p in kwargs["providers"]:
                if isinstance(p, dict):
                    p = pystac.Provider(**p)
                elif not isinstance(p, pystac.Provider):
                    continue
                providers.append(p)
            kwargs["providers"] = providers

        return (kwargs, links)
