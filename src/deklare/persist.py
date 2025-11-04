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
from typing import Callable, TypeVar, Type, NamedTuple, Self, IO
from abc import ABC, abstractmethod

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

class MediaDescription(NamedTuple):
    media_type: str
    description: str

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
        if str_src.startswith('/') and len(str_src) > 1:
            str_src = str_src[1:]
        return self.store[str_src].decode()

    def write_text(self, dest: pystac.utils.HREF, txt: str, *args, **kwargs) -> None:
        """writes the data of `txt` into the store to `dest`

        Args:
            dest : The destination to write to.
            txt : The text to write.
        """
        str_dest = str(dest)
        if str_dest.startswith('/') and len(str_dest) > 1:
            str_dest = str_dest[1:]

        self.store[str_dest] = txt.encode()

    def gen_stac_item_kwargs(self, deskriptor: dict, item_metadata: dict) -> dict:
        """generates metadata for the stac item for a deskriptor

        Args:
            deskriptor (dict): deskriptor describing
            item_metadata (dict): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            dict: _description_
        """
        if (
            "longitude" not in deskriptor or
            "start" not in deskriptor["longitude"] or
            "end" not in deskriptor["longitude"] or
            "latitude" not in deskriptor or
            "start" not in deskriptor["latitude"] or
            "end" not in deskriptor["latitude"] or
            "time" not in deskriptor or
            "variable" not in deskriptor
        ):
            raise RuntimeError("Given deskriptor does not match required metadata format")

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
                "description": self._gen_description(deskriptor),
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

    def _gen_description(self, deskriptor) -> str:
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


class DataContainer(ABC):
    @abstractmethod
    def write(self, file: IO) -> None:
        """Takes the data and writes it to the file

        Args:
            file (IO): file-like object to save data from itself too
        """
        pass

    @classmethod
    @abstractmethod
    def read(cls, file: IO) -> Self:
        """Reads the data in file and creates new data container from it

        Args:
            file (IO): file-like object containing saved data

        Returns:
            DataContainer: Container containing data saved in file
        """
        pass

    @abstractmethod
    def get_stac_metadata(self) -> dict | None:
        """optionally returns STAC metadata for the contained data

        returns:
            dict | None: dict containing STAC metadata or None for no metadata
        """
        pass

    @classmethod
    @abstractmethod
    def merge(cls, *elements: Self) -> Self:
        """merge the elements into a single DataContainer instance
        if no elements are passed return an empty container, if only one is passed acts as the identity function

        Args:
            elements (DataContainer): list of data containers to merge

        Returns:
            DataContainer: DataContainer resulting from merging elements
        """
        pass

    @abstractmethod
    def file_info(self) -> MediaDescription:
        """return media type and text describtion of file saved in `write` function
        media type should be registered in https://www.iana.org/assignments/media-types/media-types.xhtml e.g. `image/tiff`
        description should be human readable with information needed to read file

        Returns:
            MediaDescription: (media type, description)
        """
        pass


@task()
class Persister:
    def __init__(
        self,
        data_container: Type[DataContainer],
        store=None,
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
        self.data_container = data_container

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

    def configure(self, deskriptor: dict | None = None):
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

    def compute(self, data: DataContainer | None = None, **deskriptor):
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
            data = self.data_container.read(f)

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
                item_metadata = data.get_stac_metadata() or {}

                # write to file
                if isinstance(data, NodeFailedException):
                    self.store.dirfs.mkdirs("fail/", exist_ok=True)
                    with self.store.dirfs.open(
                        "fail/" + deskriptor["deskriptor_hash"], "wb"
                    ) as f:
                        data.write(f)

                else:
                    if isinstance(data, str):
                        raise RuntimeError(f"something wrong {data}")

                    try:
                        with self.store.dirfs.open(data_path, "wb") as f:
                            data.write(f)
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

        kwargs = self.stac_io.gen_stac_item_kwargs(deskriptor, item_metadata)

        item = pystac.Item(**kwargs)

        file_info = self.data_container.file_info()

        asset = pystac.Asset(
            href=f"./../../data/{deskriptor['deskriptor_hash']}",
            description=file_info.description,
            media_type=file_info.media_type,
            roles=["data"],
        )
        item.add_asset(key="data", asset=asset)

        # save in collection and avoid concurrency issues
        with self._global_lock:
            collection = pystac.Collection.from_file(
                "/stac/collection.json", self.stac_io
            )  # pretending location is absolute to stop stac from changing path
            collection.add_item(item)
            collection.save(
                catalog_type=pystac.CatalogType.SELF_CONTAINED,
                dest_href="/stac",  # pretend path is absolute so pystac doesnt try and change it
                stac_io=self.stac_io,
            )

    def _string_timestamp(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        else:
            return str(o)


@task()
class ChunkPersister:
    def __init__(
        self,
        data_container: Type[DataContainer],
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
        collection_metadata: dict = {},
        save_metadata=False,
        use_memorycache=True,
        cache = None
    ):
        """Chunks every incoming dekriptor into subchunks if deskriptor is larger than segment_slice
         or extends the deskriptor to the respective chunksize if deskriptor is smaller than segment_slice

        Args:
            store (_type_): _description_
            data_container (Type[DataContainer]): Type of DataContainer used
            dim (str, optional): _description_. Defaults to "time".
            segment_slice (dict | Callable[..., dict], optional): A dictionary containing an entry for each dimension that should be chunked. Each entry is the respective chunk size given in the units of the expected dimension of the deskriptor. For example, for a time dimension you can use pd.Timedelta. Defaults to None.
            dataset_scope (dict | Callable[...,dict], optional): The extend of the chunking. If None, the incoming deskriptor will be used as the scope. If only selected dimensions are given as dataset_scope, the scope for the other dimensions will be choosen from the incoming deskriptor. Defaults to None.
            mode (str, optional): _description_. Defaults to "overlap".
            reference (dict, optional): _description_. Defaults to None.
            force_update (bool, optional): _description_. Defaults to False.
            collection_metadata (dict, optional): Further kwargs for STAC collection. May contain keys ['id', 'title', 'keywords', 'license', 'links', 'providers']. For 'links' and 'providers' lists of either corresponding STAC objects or dicts that can be used as kwargs to construct them. Defaults to {}.
        """
        # if callable(classification_scope):
        #     self.classification_scope = classification_scope
        #     classification_scope = None
        # else:
        #     self.classification_scope = None

        self.data_container = data_container

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
                    "/stac/collection.json", self.stac_io
                )
            except:
                collection_metadata = ChunkPersister._process_collection_metadata(
                    collection_metadata
                )
                collection = pystac.Collection(**collection_metadata[0])

                for l in collection_metadata[1]:
                    collection.add_link(l)

                collection.normalize_and_save(
                    root_href="/stac",  # pretend path is absolute so pystac doesnt try and change it
                    catalog_type=pystac.CatalogType.SELF_CONTAINED,
                    stac_io=self.stac_io,
                )

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
                data_container=self.data_container,
                store=self.store,
                stac_io=self.stac_io,
                global_lock=self.mutex,
                save_metadata=self.save_metadata,
                cache=self.cache,
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

    def compute(self, *data: DataContainer | NodeFailedException, **deskriptor) -> DataContainer:
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
                    "/stac/collection.json", self.stac_io
                )  # pretend path is absolute so pystac doesnt try and change it
                collection.update_extent_from_items()
                collection.save(
                    catalog_type=pystac.CatalogType.SELF_CONTAINED,
                    dest_href="/stac",  # pretend path is absolute so pystac doesnt try and change it
                    stac_io=self.stac_io,
                )

        section = self.data_container.merge(*success)
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
