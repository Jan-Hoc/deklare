import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import (
    NodeFailedException,
    get_segments,
)

import traceback


def get_dataset_segments(
    catalog, segment_slice="60 seconds", segment_stride="60 seconds", mode="overlap"
):
    ref = {"time": pd.to_datetime("20200101")}
    mode = {"time": mode}
    segment_slice = {"time": pd.to_timedelta(segment_slice)}
    segment_stride = {"time": pd.to_timedelta(segment_stride)}
    classification_segments = []

    for catalog_item in catalog:
        if "station" not in catalog_item:
            continue

        non_sliced_segment = {
            "time": {
                "start": catalog_item["time"]["start"],
                "end": catalog_item["time"]["end"],
            },
        }

        segments = get_segments(
            non_sliced_segment,
            segment_slice,
            segment_stride,
            ref=ref,
            mode=mode,
            timestamps_as_strings=True,
            minimal_number_of_segments=1,
        )

        for segment in segments:
            segment["station"] = catalog_item["station"]
            segment["network"] = catalog_item["network"]
            segment["location"] = catalog_item["location"]
            segment["channel"] = catalog_item["channel"]

        classification_segments += segments.tolist()

    return classification_segments


class Dataset:
    def __init__(
        self,
        deskriptors,
        flows,
        transforms=None,
    ):
        self.singleton = False

        if not isinstance(flows, list):
            self.singleton = True
            flows = [flows]

        self.dataset_deskriptors = np.array(deskriptors)
        self.flows = flows

        self.indices = np.arange(len(deskriptors)).tolist()

        self.invalid_indices = {}
        self.valid_indices = {}

        self.transforms = transforms

    @property
    def deskriptors(self):
        return self.dataset_deskriptors[self.indices]

    def mask_invalid(self):
        local_dict = self.invalid_indices
        self.indices = [x for x in self.indices if x not in local_dict]

    def valid(self, idx):
        if idx in self.valid_indices:
            return True
        if idx in self.invalid_indices:
            return False

        # If we get here the idx was never tested for validity
        # so let's do it
        try:
            result = self.__getitem__(idx, only_validity=True)
            if isinstance(result, NodeFailedException):
                return False
            if isinstance(result, tuple):
                return all(
                    [not isinstance(item, NodeFailedException) for item in result]
                )
            return True
        except Exception as e:
            tqdm.write(traceback.format_exc())
            return False

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, only_validity=False):
        singleton = self.singleton

        stream_select = np.arange(len(self.flows))
        if isinstance(idx, tuple):
            idx, stream_select = idx
            if not isinstance(stream_select, (list, np.ndarray)):
                singleton = True
                stream_select = [stream_select]

        internal_idx = self.indices[idx]

        deskriptor = self.dataset_deskriptors[internal_idx]
        out = []
        for stream in stream_select:
            # check if dataset was persisted before
            values = None
            values = self.flows[stream](deskriptor=deskriptor)

            if isinstance(self.transforms, list):
                if self.transforms[stream] is not None:
                    values = self.transforms[stream](values)
            elif self.transforms is not None:
                values = self.transforms(values)

            out.append(values)

        if singleton:
            return out[0]

        return tuple(out)

    def check_validity(self, batch_size=1, num_workers=0):
        temp_transforms = self.transforms
        self.transforms = None

        this = self

        class TmpClass:
            def __getitem__(self, idx):
                return (idx, this.valid(idx))

            def __len__(self):
                return len(this)

        import torch

        for batch in tqdm(
            torch.utils.data.dataloader.DataLoader(
                TmpClass(),
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=False,
                shuffle=False,
                collate_fn=lambda x: x,
            )
        ):
            # We are updating the valid_indices and invalid_indices here in the main thread
            # and not within the possibly parallelized self.valid() calls
            for idx, valid in batch:
                if valid:
                    self.valid_indices[idx] = True
                else:
                    self.invalid_indices[idx] = True

        self.transforms = temp_transforms

    def preload(self, batch_size=1, num_workers=0):
        """Using pytorch to preload this dataset, i.e. run through the whole dataset once.
        The caching/persisting will happen inside the individual flows


        Args:
            batch_size (int, optional): batch size for loading. Defaults to 1.
            num_workers (int, optional): number of parallel workers. Defaults to 0.
        """
        temp_transforms = self.transforms
        self.transforms = None
        import torch

        for item in tqdm(
            torch.utils.data.dataloader.DataLoader(
                self,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=False,
                shuffle=False,
                collate_fn=lambda x: [],
            )
        ):
            continue

        self.transforms = temp_transforms
