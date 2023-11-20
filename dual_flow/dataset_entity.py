import sys
import torch
import logging
from torch.utils.data import Dataset, DataLoader

from utils import get_cmekg_entity_specific

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

class EntityDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        entity_lst, _, entity_type_dict = get_cmekg_entity_specific(args.data_name)

        entity_ids = []
        for entity in entity_lst:
            tmp = []

            tokens = tokenizer.tokenize(entity)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            tmp += ids
            entity_ids.append(tmp[:512])

        self.samples_entity_ids = entity_ids

    def __len__(self):
        return len(self.samples_entity_ids)

    def __getitem__(self, item):
        lst_data = {
            "entity_ids": self.samples_entity_ids[item],
        }
        return lst_data

def pack_tensor_2D(raw_lst, default, dtype, length=None):
    batch_size = len(raw_lst)
    length = length if length is not None else max(len(raw) for raw in raw_lst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, raw in enumerate(raw_lst):
        tensor[i, :len(raw)] = torch.tensor(raw, dtype=dtype)
    return tensor

def get_collate_function(tokenizer):
    def collate_function(batch):
        entity_ids_lst = [x["entity_ids"] for x in batch]
        entity_mask_lst = [len(x) * [1] for x in entity_ids_lst]

        data = {
            "entity_ids": pack_tensor_2D(entity_ids_lst, default=0, dtype=torch.int64),
            "entity_mask": pack_tensor_2D(entity_mask_lst, default=0, dtype=torch.int64),
        }

        return data
    return collate_function

def construct_data(args, per_gpu_batch_size, tokenizer, data_sampler):
    batch_size = per_gpu_batch_size * max(1, args.n_gpu)
    dataset = EntityDataset(args, tokenizer)
    sampler = data_sampler(dataset)
    collate_fn = get_collate_function(tokenizer)
    dataloader = DataLoader(dataset, sampler=sampler, 
        batch_size=batch_size, num_workers=args.data_num_workers, collate_fn=collate_fn)
    return dataset, dataloader, batch_size
