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

import datetime
from typing import Annotated, Any, Dict, Generic, TypeVar, get_args

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator
from pydantic.functional_validators import AfterValidator


def to_datetime(v: Any) -> Any:
    return pd.to_datetime(v, utc=True).tz_localize(None)


T = TypeVar("T", str, datetime.datetime, datetime.date, int, float, type(np.datetime64))
DateTimeType = Annotated[T, AfterValidator(to_datetime)]


# class DatetimeRange(BaseModel):
#     start: DateTimeType
#     end: DateTimeType


T = TypeVar("T", int, float, DateTimeType)


class Range(BaseModel, Generic[T]):
    start: T
    end: T


DatetimeRange = Range[DateTimeType]


class Deskriptor(BaseModel, validate_assignment=True):
    @model_validator(mode="before")
    def dynamic_validator(cls, values: dict[str, Any]) -> dict[str, Any]:
        kwargs = {}
        for key, value in values.items():
            if key in cls.model_fields:
                field_info = cls.model_fields[key]
                if DatetimeRange in get_args(field_info.annotation) and isinstance(
                    value, dict
                ):
                    kwargs[key] = DatetimeRange(**value)
                elif Range in get_args(field_info.annotation) and isinstance(
                    value, dict
                ):
                    kwargs[key] = Range(**value)
                else:
                    kwargs[key] = value
        return kwargs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        kwargs = {}
        for key, value in data.items():
            if key in cls.model_fields:
                field_info = cls.model_fields[key]
                if DatetimeRange in get_args(field_info.annotation) and isinstance(
                    value, dict
                ):
                    kwargs[key] = DatetimeRange(**value)
                elif Range in get_args(field_info.annotation) and isinstance(
                    value, dict
                ):
                    kwargs[key] = Range(**value)
                else:
                    kwargs[key] = value

        return cls(**kwargs)