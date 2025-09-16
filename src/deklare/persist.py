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

import json
import fsspec

from copy import copy, deepcopy
from pathlib import Path
from threading import Lock
from typing import Callable, Type

from cachetools import LRUCache

# TODO: can we implement our own hash function for deskriptors to reduce dependency on dask?
from dask.base import tokenize
from typing import Any, Iterable

import pystac
from cachetools import Cache

from .core import task
from .utils import (
    NodeFailedError,
    dict_update,
    get_segments,
)
from .data_io import StacIO, DataContainer, MediaDescription

# ToDo: Doc strings
# ToDo: more precise types


@task()
class Persister:
    def __init__(
        self,
        data_container: Type[DataContainer],
        store: fsspec.FSMap = None,
        stac_io: StacIO | None = None,
        selected_keys: Iterable | None = None,
        force_update: bool = False,
        use_memorycache: bool = True,
        cache: Cache | None = None,
        global_lock: Lock = None,
        save_metadata: bool = False,
    ) -> None:
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

    def configure(self, deskriptor: dict | None = None) -> dict:
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
            if deskriptor["self"].get("use_memorycache", True) and data_path in self.cache:
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
                if data_path in self.cache:
                    cached = self.cache[data_path]
                    return cached
        if deskriptor["action"] == "load" or deskriptor["action"] == "load_from_cache":
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
                if isinstance(data, NodeFailedError):
                    self.store.dirfs.mkdirs("fail/", exist_ok=True)
                    with self.store.dirfs.open("fail/" + deskriptor["deskriptor_hash"], "wb") as f:
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
                        self._save_metadata(deskriptor, item_metadata, data.file_info())

            except Exception as e:
                print("Error during Persister", repr(e))

            return data
        else:
            raise NodeFailedError("A bug in Persister. Please report.")

    def is_valid(self, deskriptor: dict) -> bool | None:
        """Checks if persisted object for `deskriptor`
        exists and is valid (i.e. is not of type NodeFailedException).

        Args:
            deskriptor (dict): The deskriptor that should be checked

        Returns:
            boolean | None: Returns false if the persisted item is of type NodeFailedException
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
            str: hash of the request
        """
        r = {k: v for k, v in deskriptor.items() if k != "self"}
        s = json.dumps(r, sort_keys=True, skipkeys=True, default=_string_timestamp)
        deskriptor_hash = tokenize(s)

        return deskriptor_hash

    def _save_metadata(self, deskriptor: dict, item_metadata: dict, file_info: MediaDescription) -> None:
        """saves metadata for given chunk using STAC (https://stacspec.org/)

        Args:
            deskriptor (dict): the deskriptor containing the temporal and spacial boundaries
            item_metadata (dict): additional metadata passed by loader to save in STAC item
            file_info (MediaDescription): information about the saved file
        """

        kwargs = self.stac_io.gen_stac_item_kwargs(deskriptor, item_metadata)

        item = pystac.Item(**kwargs)

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


def _string_timestamp(o: object) -> str:
    if hasattr(o, "isoformat"):
        return o.isoformat()
    else:
        return str(o)


@task()
class ChunkPersister:
    def __init__(
        self,
        data_container: Type[DataContainer],
        store: fsspec.FSMap | None = None,
        filesystem: fsspec.AbstractFileSystem | None = None,
        dim: str = "time",
        segment_slice: dict | Callable[..., dict] = None,
        segment_stride: dict | Callable[..., dict] = None,
        dataset_scope: dict | Callable[..., dict] = None,
        mode: str = "overlap",
        reference: dict = None,
        force_update: bool = False,
        merge_function: Callable | None = None,
        collection_metadata: dict | None = None,
        save_metadata: bool = False,
        use_memorycache: bool = True,
        cache: Cache | None = None,
    ) -> None:
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
        self.data_container = data_container

        self.use_memorycache = use_memorycache
        self.cache = cache or LRUCache(10)

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

        self.store = store or self.filesystem.get_mapper()

        self.stac_io = StacIO(store=store)

        if self.save_metadata:
            # create collection if doesn't exist
            try:
                collection = pystac.Collection.from_file("/stac/collection.json", self.stac_io)
            except Exception:
                collection_metadata = ChunkPersister._process_collection_metadata(collection_metadata or {})
                collection = pystac.Collection(**collection_metadata[0])

                for link in collection_metadata[1]:
                    collection.add_link(link)

                collection.normalize_and_save(
                    root_href="/stac",  # pretend path is absolute so pystac doesnt try and change it
                    catalog_type=pystac.CatalogType.SELF_CONTAINED,
                    stac_io=self.stac_io,
                )

        self.mutex = Lock()

    def __dask_tokenize__(self) -> tuple:
        return (ChunkPersister,)

    def configure(self, deskriptor: dict | None = None) -> dict:
        rs = deskriptor["self"]
        if rs.get("bypass", False):
            return deskriptor

        def get_value(attr_name: str) -> Any:  # noqa: ANN401
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
                use_memorycache=self.use_memorycache,
            )
            cloned_persister.dask_key_name = self.dask_key_name + "_persister"
            dict_update(
                segment_deskriptor,
                {
                    "config": {
                        "keys": {self.dask_key_name + "_persister": {"force_update": rs.get("force_update", False)}}
                    }
                },
            )
            cloned_persisters += [cloned_persister.compute]

        # Insert predecessor
        # new_deskriptor = {}
        deskriptor["clone_dependencies"] = cloned_deskriptors
        deskriptor["insert_predecessor"] = cloned_persisters

        return deskriptor

    def compute(self, *data: DataContainer | NodeFailedError, **deskriptor) -> DataContainer:
        def unpack_list(inputlist):
            new_list = []
            for item in inputlist:
                if isinstance(item, (tuple, list)):
                    new_list += unpack_list(item)
                else:
                    new_list += [item]
            return new_list

        data = unpack_list(data)
        success = [d for d in data if not isinstance(d, NodeFailedError)]

        if not success:
            failed = [str(d) for d in data if isinstance(d, NodeFailedError)]
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

    @staticmethod
    def _process_collection_metadata(collection_metadata: dict | None = None) -> tuple[dict, list]:
        """return dict with arguments to create pystac.Collection

        Args:
            collection_metadata (dict, optional): Dict containing keyword arguments for pystac.Collection constructor

        Returns:
            (dict, list): dict with list of keyword arguments, list of pystac.Link objects to add to collection
        """

        collection_metadata = collection_metadata or {}

        links = []
        for link in collection_metadata.get("links", []):
            if isinstance(link, dict):
                link = pystac.Link(**link)
            elif not isinstance(link, pystac.Link):
                continue
            links.append(link)

        kwargs = copy(DEFAULT_KWARGS)
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


DEFAULT_KWARGS = {
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
