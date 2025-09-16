import traceback
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from .utils import NodeFailedError

# ToDo: Doc strings
# ToDo: more precise types


# ToDo: class attributes
class Dataset:
    def __init__(
        self,
        deskriptors: list[dict],
        flows,  # ToDo: typing # noqa: ANN001
        transforms: list[Callable] | Callable | None = None,
    ) -> None:
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
    def deskriptors(self) -> dict:
        return self.dataset_deskriptors[self.indices]

    def mask_invalid(self) -> None:
        local_dict = self.invalid_indices
        self.indices = [x for x in self.indices if x not in local_dict]

    # ToDo: check if this needs to be so complicated
    def valid(self, idx: int) -> bool:
        if idx in self.valid_indices:
            return True
        if idx in self.invalid_indices:
            return False

        # If we get here the idx was never tested for validity
        # so let's do it
        try:
            result = self.__getitem__(idx, only_validity=True)
            if isinstance(result, NodeFailedError):
                return False
            if isinstance(result, tuple):
                return all([not isinstance(item, NodeFailedError) for item in result])
            return True
        except Exception:
            tqdm.write(traceback.format_exc())
            return False

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
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
            values = self.flows[stream].query(deskriptor)

            if isinstance(self.transforms, list):
                if self.transforms[stream] is not None:
                    values = self.transforms[stream](values)
            elif self.transforms is not None:
                values = self.transforms(values)

            out.append(values)

        if singleton:
            return out[0]

        return tuple(out)

    def check_validity(self, batch_size: int = 1, num_workers: int = 0) -> None:
        temp_transforms = self.transforms
        self.transforms = None

        this = self

        class TmpClass:
            def __getitem__(self, idx: int) -> tuple:
                return (idx, this.valid(idx))

            def __len__(self) -> int:
                return len(this)

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

    def preload(self, batch_size: int = 1, num_workers: int = 0) -> None:
        """Using pytorch to preload this dataset, i.e. run through the whole dataset once.
        The caching/persisting will happen inside the individual flows


        Args:
            batch_size (int, optional): batch size for loading. Defaults to 1.
            num_workers (int, optional): number of parallel workers. Defaults to 0.
        """
        temp_transforms = self.transforms
        self.transforms = None

        for _ in tqdm(
            torch.utils.data.dataloader.DataLoader(
                self,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=False,
                shuffle=False,
                collate_fn=lambda _: [],
            )
        ):
            continue

        self.transforms = temp_transforms
