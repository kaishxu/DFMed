import sys
import copy
import torch
import pickle
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils import get_cmekg_entity_specific

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

def load_data(data_name, data_type):
    with open(f"../data/{data_name}_{data_type}_kg_entity_specific.pk", "rb") as f:
        data = pickle.load(f)

    logger.info(f"Num of {data_name} {data_type} dialogues: %d", len(data))
    return data

def get_turn_range(raw_ids):
    length_lst = [len(x) for x in raw_ids]  #每个turn的长度

    turn_range = []
    end = 1
    while end < len(raw_ids):
        for start in range(0, end):
            if np.sum(length_lst[start:end]) <= 512:
                break
        turn_range.append([start, end])
        end += 2
    return turn_range

def get_samples(raw_ids, turn_range, start_token=None, end_token=None, is_target=False):
    ids = []
    for (start, end) in turn_range:
        ids_tmp = []  #一个样本
        if not is_target:  #构建context样本
            for i in range(start, end):
                ids_tmp += raw_ids[i]
            if len(ids_tmp) == 512:  #为补[CLS]留位置
                ids_tmp = ids_tmp[0:1] + ids_tmp[2:]
            ids_tmp = [start_token] + ids_tmp

        else:  #构建target样本
            ids_tmp += raw_ids[end]
            ids_tmp[0] = start_token
            ids_tmp = ids_tmp[:512-1]
            ids_tmp += [end_token]

        assert len(ids_tmp) <= 512
        ids.append(ids_tmp)
    return ids  #多个样本

def process_data(args,
                mode,
                data_name,
                data_type,
                raw_data,
                tokenizer):

    act_id = {"INQUIRE": 0, "DIAGNOSIS": 1, "TREATMENT": 2, "TEST": 3, "PRECAUTION": 4, "INFORM": 5, "CHITCHAT": 6}

    if not args.for_meddg_160:
        #retrieved kg entity
        with open(f"{args.act_entity_dir}/{data_name}_{data_type}_ranked_entity.pk", "rb") as f:
            entity_ranked_dict = pickle.load(f)

        #cmekg entity
        entity_lst, _ = get_cmekg_entity_specific(data_name)

    #predicted act and selected entity
    if mode != "train":

        with open(f"{args.act_entity_dir}/{data_name}_{data_type}_predicted_act.pk", "rb") as f:
            _, _, _, act_predicted_dict = pickle.load(f)

        if args.for_meddg_160:
            with open(f"{args.act_entity_dir}/{data_name}_{data_type}_selected_entity.pk", "rb") as f:
                _, _, _, entity_predicted_dict = pickle.load(f)

    #role dict
    role_token_id = {"patient": 1, "doctor": 2}

    samples_context_ids = []
    samples_context_raw = []
    samples_target_ids = []
    samples_target_raw = []
    sample_entity_ids = []
    samples_idx = []
    samples_turn_idx = []
    for sample in tqdm(raw_data, desc=f"Construct {data_type} samples"):

        dialogue_history = sample["dialogues"]
        if dialogue_history[-1]["role"] == "patient":  #所有对话以doctor回复结束
            del dialogue_history[-1]

        assert len(dialogue_history) % 2 == 0

        context_ids_tmp = []
        for i, turn in enumerate(dialogue_history):
            tmp = [role_token_id[turn["role"]]]
            tokens = tokenizer.tokenize(turn["sentence"])
            ids = tokenizer.convert_tokens_to_ids(tokens)
            tmp += ids  #dialogue input ids

            extra_length = max(len(tmp) - 512, 0)  #确认超出长度
            if extra_length > 0 and turn["role"] == "patient":
                tmp = tmp[:-extra_length]

            context_ids_tmp.append(tmp)
            if turn["role"] == "doctor":
                #保存原始target
                samples_target_raw.append(turn["sentence"])

                #保存原始context
                context_raw_tmp = []
                for j in range(i):
                    context_raw_tmp.append(dialogue_history[j]["sentence"])
                samples_context_raw.append({
                                        "context": context_raw_tmp,
                                        "idx": sample["idx"],
                                        "turn_idx": str(i),
                                        })

                tmp_entity = [tokenizer.cls_token_id]

                if mode != "train":
                    for x in act_predicted_dict[sample["idx"] + "_" + str(turn["turn"])]:
                        tmp_entity += [act_id[x]+1,]

                    if args.for_meddg_160:
                        for x in entity_predicted_dict[sample["idx"] + "_" + str(turn["turn"])]:
                            tokens = tokenizer.tokenize(x)
                            ids = tokenizer.convert_tokens_to_ids(tokens)
                            tmp_entity += ids
                else:
                    turn_act_lst = set()
                    for act_lst in turn["act"]:
                        turn_act_lst.update(act_lst)
                    turn_act_lst = [act_id[x]+1 for x in act_id if x in turn_act_lst]  #直接保存act编号
                    tmp_entity += turn_act_lst

                    if args.for_meddg_160:    
                        for entity in turn["entity"]:
                            tokens = tokenizer.tokenize(entity)
                            ids = tokenizer.convert_tokens_to_ids(tokens)
                            tmp_entity += ids  #entity ids

                if not args.for_meddg_160:    
                    for idx in entity_ranked_dict[sample["idx"] + "_" + str(turn["turn"])][:args.k_entity]:
                        tokens = tokenizer.tokenize(entity_lst[idx])
                        ids = tokenizer.convert_tokens_to_ids(tokens)
                        tmp_entity += [tokenizer.sep_token_id] + ids #entity ids

                tmp_entity = tmp_entity[:512]
                sample_entity_ids.append(tmp_entity)

        turn_range = get_turn_range(context_ids_tmp)  #包含多个样本
        context_ids = get_samples(context_ids_tmp, turn_range, start_token=tokenizer.cls_token_id)  #包含多个样本
        target_ids = get_samples(context_ids_tmp, turn_range, start_token=tokenizer.cls_token_id, end_token=tokenizer.sep_token_id, is_target=True)  #包含多个样本

        samples_context_ids += context_ids
        samples_target_ids += target_ids

        turn_idx = [str(end) for (start, end) in turn_range]
        samples_turn_idx += turn_idx
        samples_idx += [sample["idx"]] * len(context_ids)

    return (samples_context_ids, samples_target_ids, samples_target_raw, samples_context_raw, sample_entity_ids,
            samples_idx, samples_turn_idx)

class BaseDataset(Dataset):
    def __init__(self, args, data_name, data_type, mode, tokenizer):
        self.data_name = data_name
        self.data_type = data_type
        self.mode = mode
        self.tokenizer = tokenizer
        self.raw_data = load_data(data_name, data_type)
        (self.samples_context_ids, self.samples_target_ids, self.samples_target_raw, self.samples_context_raw, self.sample_entity_ids,
        self.samples_idx, self.samples_turn_idx) = process_data(args, mode, data_name, data_type, self.raw_data, tokenizer)

    def __len__(self):
        assert len(self.samples_context_ids) == len(self.samples_target_ids)
        assert len(self.samples_idx) == len(self.samples_turn_idx)
        assert len(self.samples_target_raw) == len(self.samples_idx)
        assert len(self.samples_context_raw) == len(self.samples_target_raw)
        assert len(self.samples_context_ids) == len(self.sample_entity_ids)
        return len(self.samples_context_ids)

    def __getitem__(self, item):
        lst_data = {
            "context_ids": self.samples_context_ids[item],
            "target_ids": self.samples_target_ids[item],
            "entity_ids": self.sample_entity_ids[item],
            "idx": self.samples_idx[item],
            "turn_idx": self.samples_turn_idx[item],
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
        context_ids_lst = [x["context_ids"] for x in batch]
        context_mask_lst = [[1] * len(context_ids) for context_ids in context_ids_lst]
        target_ids_lst = [x["target_ids"] for x in batch]
        target_mask_lst = [[1] * len(target_ids) for target_ids in target_ids_lst]
        entity_ids_lst = [x["entity_ids"] for x in batch]
        entity_mask_lst = [[1] * len(entity_ids) for entity_ids in entity_ids_lst]
        labels_lst = [x["target_ids"][1:] + [-100] for x in batch]  #补"-100"从而保持跟target的长度一致

        data = {
            "input_ids": pack_tensor_2D(context_ids_lst, default=0, dtype=torch.int64),
            "attention_mask": pack_tensor_2D(context_mask_lst, default=0, dtype=torch.int64),
            "decoder_input_ids": pack_tensor_2D(target_ids_lst, default=0, dtype=torch.int64),
            "decoder_attention_mask": pack_tensor_2D(target_mask_lst, default=0, dtype=torch.int64),
            "entity_input_ids": pack_tensor_2D(entity_ids_lst, default=0, dtype=torch.int64),
            "entity_attention_mask": pack_tensor_2D(entity_mask_lst, default=0, dtype=torch.int64),
            "labels": pack_tensor_2D(labels_lst, default=-100, dtype=torch.int64),
        }

        idx = [x["idx"] for x in batch]
        turn_idx = [x["turn_idx"] for x in batch]
        return data, idx, turn_idx
    return collate_function

def construct_data(args, data_type, mode, per_gpu_batch_size, tokenizer, data_sampler):
    batch_size = per_gpu_batch_size * max(1, args.n_gpu)
    dataset = BaseDataset(args, args.data_name, data_type, mode, tokenizer)
    sampler = data_sampler(dataset)
    collate_fn = get_collate_function(tokenizer)
    dataloader = DataLoader(dataset, sampler=sampler, 
        batch_size=batch_size, num_workers=args.data_num_workers, collate_fn=collate_fn)
    return dataset, dataloader, batch_size
