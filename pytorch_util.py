from typing import Any, Callable

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


class GetBatchFunDatasetWrapper(Dataset):
    def __init__(self, getbatch_fun) -> None:
        super().__init__()
        self.get_batch_fun = getbatch_fun

    def __getitem__(self, index):
        return self.get_batch_fun(index)


class MessagingSampler(Sampler):
    def __init__(self, message_supplier):
        super().__init__(None)
        self.message_supplier = message_supplier

    def __iter__(self):
        while True:
            yield [self.message_supplier()]


def build_BatchingDataLoader(
    get_batch_fun: Callable[[], Any],
    message_supplier: Callable[[], Any] = lambda: None,
    collate_fn=lambda x: x[0],
):
    sampler = MessagingSampler(message_supplier=message_supplier)
    data_loader = torch.utils.data.DataLoader(
        dataset=GetBatchFunDatasetWrapper(get_batch_fun),
        num_workers=0,
        collate_fn=collate_fn,
        batch_sampler=sampler,
    )
    return data_loader
