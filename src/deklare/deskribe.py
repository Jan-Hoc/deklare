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

from __future__ import annotations

import datetime
from typing import Annotated, Any, Generic, TypeVar, get_args

import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import DatetimeScalar
from pydantic import BaseModel, Field, model_validator
from pydantic.functional_validators import AfterValidator

# ToDo: fix deskriptor types (then also in doc strings)


def _to_datetime(v: DatetimeScalar) -> pd.Timestamp:
    """helper function to create DateTimeType"""
    return pd.to_datetime(v, utc=True).tz_localize(None)


DatetimeT = TypeVar("DatetimeT", int, float, str, datetime.date, datetime.datetime, np.datetime64)


DateTimeType = Annotated[DatetimeT, AfterValidator(_to_datetime)]


T = TypeVar("T", int, float, DateTimeType)


class Range(BaseModel, Generic[T]):
    start: T
    end: T


DatetimeRange = Range[DateTimeType]


def _transform_to_nested(input_dict: dict, separator: str = ".") -> dict:
    """transform the flat JSON keys with dots (split) into nested JSON keys

    Args:
        input_dict (dict): dict with possibly flattened JSON keys
        separator (str): seperator used to flatten keys
    """
    transformed = {}
    for key, value in input_dict.items():
        parts = key.split(separator)
        current = transformed
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return transformed


# ToDo: fix deskriptor class to also accept dicts
class Deskriptor(BaseModel, validate_assignment=True):
    """Deskriptor to represent queries for data

    Attributes:
        config (dict[str, Any]): config of deskriptor
    """

    config: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    @model_validator(mode="before")
    def dynamic_validator(cls, values: dict[str, Any]) -> dict[str, Any]:
        """validate and transform input data before model instantiation
        converts nested dictionaries into `DatetimeRange` or `Range` instances if required

        Args:
            values (dict[str, Any]): raw input values to validate

        Returns:
            dict[str, Any]: validated data
        """

        kwargs = {}
        values = _transform_to_nested(values)

        for key, value in values.items():
            if key in cls.model_fields:
                field_info = cls.model_fields[key]
                if DatetimeRange in get_args(field_info.annotation) and isinstance(value, dict):
                    kwargs[key] = DatetimeRange(**value)
                elif Range in get_args(field_info.annotation) and isinstance(value, dict):
                    kwargs[key] = Range(**value)
                else:
                    kwargs[key] = value

        return kwargs

    def to_dict(self, *, remove_none: bool = True) -> dict[str, Any]:
        """transforms instance into dict representation

        Args:
            remove_none (bool): ignores None values if True. defaults to True

        Returns:
            dict[str, Any]: dictionary containing data in instance
        """
        result = {}
        for field_name, field_value in self:
            if remove_none and field_value is None:
                continue
            elif isinstance(field_value, (DatetimeRange, Range)):
                result[field_name] = dict(field_value)
            else:
                result[field_name] = field_value
        return result

    def update(self, other: Deskriptor) -> None:
        """updates instance given other deskriptor

        Args:
            other (Deskriptor): deskriptor to use to update instance
        """
        self.config.update(other.config)

    def get(self, key: str, default: Any) -> Any:  # noqa: ANN401
        return self.config.get(key, default)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Deskriptor:
        """create deskriptor from data

        Args:
            data (dict[str, Any]): data to use to create deskriptor

        Returns:
            Deskriptor: containing data
        """
        kwargs = {}

        data = _transform_to_nested(data)

        for key, value in data.items():
            if key in cls.model_fields:
                field_info = cls.model_fields[key]
                if DatetimeRange in get_args(field_info.annotation) and isinstance(value, dict):
                    kwargs[key] = DatetimeRange(**value)
                elif Range in get_args(field_info.annotation) and isinstance(value, dict):
                    kwargs[key] = Range(**value)
                else:
                    kwargs[key] = value

        return cls(**kwargs)

    @staticmethod
    def update_from_config_dict(deskriptor: dict, config: dict) -> dict:
        """checks if deskriptor has config defaults defined in deskriptor['config']['config']
        replaces values in config if not present in deskriptor

        Args:
            deskriptor (dict): original deskriptor
            config (dict): config with default values

        Returns:
            dict: updated deskriptor
        """
        if "config" not in deskriptor:
            return deskriptor

        config_keys = deskriptor["config"]

        if "config" not in config_keys:
            return deskriptor

        config_keys = config_keys["config"]

        for key, entry in config_keys.items():
            if key not in config:
                raise KeyError(f"Invalid config key '{key}' (not present)")
            if entry not in config[key]:
                raise KeyError(f"Invalid config key '{key}'/'{entry}' (not present)")

            # dont replace if already in deskriptor
            if key not in deskriptor:
                deskriptor[key] = config[key][entry]

        return deskriptor
