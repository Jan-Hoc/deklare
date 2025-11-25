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

import pystac
import xarray as xr
from datetime import datetime, timezone
from typing import NamedTuple, Self, IO
from abc import ABC, abstractmethod
from shapely.geometry import Polygon, mapping


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
        if str_src.startswith("/") and len(str_src) > 1:
            str_src = str_src[1:]
        return self.store[str_src].decode()

    def write_text(self, dest: pystac.utils.HREF, txt: str, *args, **kwargs) -> None:
        """writes the data of `txt` into the store to `dest`

        Args:
            dest : The destination to write to.
            txt : The text to write.
        """
        str_dest = str(dest)
        if str_dest.startswith("/") and len(str_dest) > 1:
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
            "longitude" not in deskriptor
            or "start" not in deskriptor["longitude"]
            or "end" not in deskriptor["longitude"]
            or "latitude" not in deskriptor
            or "start" not in deskriptor["latitude"]
            or "end" not in deskriptor["latitude"]
            or "time" not in deskriptor
            or "variable" not in deskriptor
        ):
            raise RuntimeError(
                "Given deskriptor does not match required metadata format"
            )

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


class XArrayContainer(DataContainer):
    data: xr.Dataset

    def __init__(self, data: xr.Dataset):
        super().__init__()
        self.data = data.copy()

    def __getitem__(self, index) -> xr.DataArray | xr.Dataset:
        """function to make XArrayContainer indexable

        Args:
            index: index to retrieve data from container

        Returns:
            xr.DataArray | xr.Dataset: data corresponding to index
        """
        return self.data[index]

    def write(self, file: IO) -> None:
        """Takes the data and writes it to the file

        Args:
            file (IO): file-like object to save data from itself too
        """
        self.data.to_netcdf(file)

    @classmethod
    def read(cls, file: IO) -> Self:
        """Reads the data in file and creates new data container from it

        Args:
            file (IO): file-like object containing saved data

        Returns:
            XArrayContainer: Container containing data saved in file
        """
        return cls(xr.open_dataset(file))

    def get_stac_metadata(self) -> dict | None:
        """optionally returns STAC metadata for the contained data

        returns:
            dict | None: dict containing STAC metadata or None for no metadata
        """
        bbox = None
        if {"lat", "lon"}.issubset(self.data):
            lats = self.data["lat"].values
            lons = self.data["lon"].values
            bbox = [
                float(lons.min()),
                float(lats.min()),
                float(lons.max()),
                float(lats.max()),
            ]

        time = None
        if "time" in self.data.coords:
            t0 = self.data["time"].values[0]
            ts = t0.astype("datetime64[ns]").astype("int64") / 1e9
            time = datetime.fromtimestamp(ts, timezone.utc).isoformat()

        item = {
            "type": "Feature",
            "bbox": bbox,
            "geometry": None
            if bbox is None
            else {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]
                ],
            },
            "properties": {
                "datetime": time,
            },
            "links": [],
        }

        return item

    @classmethod
    def merge(cls, *elements: Self) -> Self:
        """merge the elements into a single XArrayContainer instance
        if no elements are passed return an empty container, if only one is passed acts as the identity function

        Args:
            elements (XArrayContainer): list of xarray containers to merge

        Returns:
            XArrayContainer: XArrayContainer resulting from merging elements
        """
        ds = xr.merge([e.data for e in elements])

        return cls(ds)

    def file_info(self) -> MediaDescription:
        """returns media type and text description of file saved in `write` function, which is netCDF file
        media type should be registered in https://www.iana.org/assignments/media-types/media-types.xhtml e.g. `image/tiff`
        description should be human readable with information needed to read file

        Returns:
            MediaDescription: (media type, description)
        """
        return MediaDescription(
            "application/octet-stream",
            "NetCDF of Data",
        )
