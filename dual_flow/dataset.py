import sys
import copy
import torch
import random
import pickle
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils import squeeze_lst, get_ner_entity, get_cmekg_entity_specific

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
    length_lst = [len(x) for x in raw_ids]  #length of each turn

    turn_range = []
    end = 1
    while end < len(raw_ids):
        for start in range(0, end):
            if np.sum(length_lst[start:end]) <= 512:
                break
        turn_range.append([start, end])
        end += 2
    return turn_range

def get_context_samples(raw_ids, turn_range, start_token=None):
    ids = []
    idx = []
    for (start, end) in turn_range:
        ids_tmp = [start_token]  #one sample
        idx_tmp = []
        for i in range(start, end):
            c_start = len(ids_tmp)
            ids_tmp += raw_ids[i]
            c_end = len(ids_tmp)
            idx_tmp.append((c_start, c_end))

        if len(ids_tmp) > 512:
            ids_tmp = ids_tmp[:512]
            idx_tmp[-1] = (idx_tmp[-1][0], 512)

        assert len(ids_tmp) <= 512
        assert len(ids_tmp) == idx_tmp[-1][1]

        ids.append(ids_tmp)
        idx.append(idx_tmp)
    return ids, idx  #multiple samples

def process_data(data_name,
                data_type,
                mode,
                raw_data,
                entity_lst,
                entity_dict,
                tokenizer):

    #ner entity
    _, ner_entity_dict = get_ner_entity(data_name)

    #sub kg
    with open(f"../data/{data_name}_{data_type}_sub_kg.pk", "rb") as f:
        sub_kg_dict = pickle.load(f)
    with open(f"../data/{data_name}_{data_type}_sub_kg_center.pk", "rb") as f:
        center_node_dict = pickle.load(f)

    #role dict
    role_token_id = {"patient": 1, "doctor": 2}

    #act dict
    act_id = {"INQUIRE": 0, "DIAGNOSIS": 1, "TREATMENT": 2, "TEST": 3, "PRECAUTION": 4, "INFORM": 5, "CHITCHAT": 6}

    samples_context_ids = []
    samples_context_idx = []
    samples_context_raw = []
    samples_target_raw = []

    samples_act_turn_idx = []
    samples_entity_turn_idx = []

    samples_act_labels = []
    samples_entity_labels = []
    samples_target_kg_entity_idx = []
    samples_neg_kg_entity_idx = []

    samples_idx = []
    samples_turn_idx = []
    for sample in tqdm(raw_data, desc=f"Construct {data_type} samples"):

        dialogue_history = sample["dialogues"]
        if dialogue_history[-1]["role"] == "patient":  #all conversations end with doctor utterances
            del dialogue_history[-1]

        assert len(dialogue_history) % 2 == 0

        context_ids_tmp = []
        act_raw = []
        entity_raw = []
        for i, turn in enumerate(dialogue_history):
            context_ids_turn_tmp = [role_token_id[turn["role"]]]  #add role token
            tokens = tokenizer.tokenize(turn["sentence"])
            ids = tokenizer.convert_tokens_to_ids(tokens)
            context_ids_turn_tmp += ids  #dialogue input ids

            extra_length = max(len(context_ids_turn_tmp) - 512, 0)  #confirm exceeding length
            if extra_length > 0 and turn["role"] == "patient":
                context_ids_turn_tmp = context_ids_turn_tmp[:-extra_length]
            context_ids_tmp.append(context_ids_turn_tmp)

            #save raw entity
            entity_raw.append([entity_dict[x] for x in turn["sub_kg_entity"]])  #the entity of the current turn and the entity of sub-kg
            assert len(turn["sub_kg_entity"]) == len(set(turn["sub_kg_entity"]))

            #save raw act
            turn_act_lst = set()
            for act_lst in turn["act"]:
                turn_act_lst.update(act_lst)
            turn_act_lst = [act_id[act] for act in act_id if act in turn_act_lst]
            act_raw.append(turn_act_lst)

            if turn["role"] == "doctor":
                #save original target
                samples_target_raw.append(turn["sentence"])

                #save original context
                context_raw_tmp = []
                for j in range(i):
                    context_raw_tmp.append(dialogue_history[j]["sentence"])
                samples_context_raw.append({
                                        "idx": sample["idx"],
                                        "turn_idx": str(i),
                                        "context": context_raw_tmp,
                                        "entity": turn["entity"],
                                        "target_kg_entity": turn["target_kg_entity"],
                                        "response": turn["sentence"],
                                        })

                sub_kg = sub_kg_dict[sample["idx"] + "_" + str(turn["turn"])]
                center_node = center_node_dict[sample["idx"] + "_" + str(turn["turn"])]

                #save target/negative kg entity
                target_kg_entity_lst = copy.deepcopy(turn["target_kg_entity"])
                target_kg_entity_lst = [x[0] for x in target_kg_entity_lst]
                neg_tmp = [x for x in sub_kg if x not in center_node and x not in target_kg_entity_lst]  #in sub-kg, entities that do not belong to the central node and target node are negative.
                neg_tmp.sort()  #sort it in order for easy reproduction
                neg_n = 20
                if len(target_kg_entity_lst) > 0:
                    if len(neg_tmp) >= neg_n:
                        neg_kg_entity_lst = random.sample(neg_tmp, neg_n)
                    else:
                        neg_kg_entity_lst = neg_tmp + random.sample(entity_lst, neg_n - len(neg_tmp))
                else:
                    neg_kg_entity_lst = []

                kg_entity_idx = squeeze_lst([entity_dict[x] for x in target_kg_entity_lst])
                samples_target_kg_entity_idx.append(copy.deepcopy(kg_entity_idx))
                kg_entity_idx = [entity_dict[x] for x in neg_kg_entity_lst]
                samples_neg_kg_entity_idx.append(copy.deepcopy(kg_entity_idx))

                #act label
                act_labels = [0] * 7
                for act in turn_act_lst:
                    act_labels[act] = 1
                samples_act_labels.append(act_labels)

                #entity label
                entity_labels = [0] * len(ner_entity_dict)
                for entity in turn["entity"]:
                    entity_labels[ner_entity_dict[entity]] = 1
                samples_entity_labels.append(entity_labels)

        turn_range = get_turn_range(context_ids_tmp)  #get the context turn range that meets the encoding length requirements (<=512)
        context_ids, context_idx = get_context_samples(context_ids_tmp, turn_range, start_token=tokenizer.cls_token_id)  #contains multiple samples

        samples_context_ids += context_ids
        samples_context_idx += context_idx

        for i, (start, end) in enumerate(turn_range):
            act_turn_idx = act_raw[start:end]
            entity_turn_idx = entity_raw[start:end]
            samples_act_turn_idx.append(copy.deepcopy(act_turn_idx))
            samples_entity_turn_idx.append(copy.deepcopy(entity_turn_idx))

        turn_idx = [str(end) for (start, end) in turn_range]
        samples_turn_idx += turn_idx
        samples_idx += [sample["idx"]] * len(context_ids)

    return (samples_context_ids, samples_context_idx,
            samples_act_turn_idx, samples_entity_turn_idx,
            samples_act_labels, samples_entity_labels,
            samples_target_kg_entity_idx, samples_neg_kg_entity_idx,
            samples_target_raw, samples_context_raw,
            samples_idx, samples_turn_idx)

class BaseDataset(Dataset):
    def __init__(self, data_name, data_type, mode, tokenizer):
        self.data_name = data_name
        self.data_type = data_type
        self.mode = mode
        self.tokenizer = tokenizer
        self.raw_data = load_data(data_name, data_type)
        self.entity_lst, self.entity_dict, _ = get_cmekg_entity_specific(data_name)
        (self.samples_context_ids, self.samples_context_idx,
        self.samples_act_turn_idx, self.samples_entity_turn_idx,
        self.samples_act_labels, self.samples_entity_labels,
        self.samples_target_kg_entity_idx, self.samples_neg_kg_entity_idx,
        self.samples_target_raw, self.samples_context_raw,
        self.samples_idx, self.samples_turn_idx)= process_data(data_name, data_type, mode, self.raw_data, self.entity_lst, self.entity_dict, tokenizer)

    def __len__(self):
        assert len(self.samples_context_ids) == len(self.samples_context_idx)
        assert len(self.samples_idx) == len(self.samples_turn_idx)
        assert len(self.samples_context_raw) == len(self.samples_target_raw)
        return len(self.samples_context_ids)

    def __getitem__(self, item):
        lst_data = {
            "context_ids": self.samples_context_ids[item],
            "context_idx": self.samples_context_idx[item],
            "act_turn_idx": self.samples_act_turn_idx[item],
            "entity_turn_idx": self.samples_entity_turn_idx[item],
            "idx": self.samples_idx[item],
            "turn_idx": self.samples_turn_idx[item],
            "act_labels": self.samples_act_labels[item],
            "entity_labels": self.samples_entity_labels[item],
            "target_kg_entity_idx": self.samples_target_kg_entity_idx[item],
            "neg_kg_entity_idx": self.samples_neg_kg_entity_idx[item],
        }
        return lst_data

def pack_tensor_2D(raw_lst, default, dtype, length=None):
    batch_size = len(raw_lst)
    length = length if length is not None else max(len(raw) for raw in raw_lst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, raw in enumerate(raw_lst):
        tensor[i, :len(raw)] = torch.tensor(raw, dtype=dtype)
    return tensor

def get_collate_function(data_name):
    def collate_function(batch):
        context_ids_lst = [x["context_ids"] for x in batch]
        context_idx_lst = [x["context_idx"] for x in batch]
        context_mask_lst = [[1] * len(context_ids) for context_ids in context_ids_lst]

        act_turn_idx_lst = [x["act_turn_idx"] for x in batch]
        entity_turn_idx_lst = [x["entity_turn_idx"] for x in batch]

        # collect all entity idx in batch
        index = 0
        batch_entity_turn_idx_lst = []
        batch_entity_turn_idx_dict = {}
        tmp_idx_lst = set()
        for i in range(len(entity_turn_idx_lst)):
            for k in range(len(entity_turn_idx_lst[i])):
                for item in entity_turn_idx_lst[i][k]:
                    if item not in tmp_idx_lst:
                        batch_entity_turn_idx_lst.append(item)
                        batch_entity_turn_idx_dict[item] = index
                        tmp_idx_lst.add(item)
                        index += 1

        # replace entity idx in batch with new idx
        new_entity_turn_idx_lst = []
        for i in range(len(entity_turn_idx_lst)):
            new_entity_turn_idx_lst_i = []
            for k in range(len(entity_turn_idx_lst[i])):
                new_entity_turn_idx_lst_k = []
                for item in entity_turn_idx_lst[i][k]:
                    new_entity_turn_idx_lst_k.append(batch_entity_turn_idx_dict[item])
                new_entity_turn_idx_lst_i.append(new_entity_turn_idx_lst_k)
            new_entity_turn_idx_lst.append(new_entity_turn_idx_lst_i)

        act_labels_lst = [x["act_labels"] for x in batch]
        entity_labels_lst = [x["entity_labels"] for x in batch]

        target_kg_entity_idx_lst = [x["target_kg_entity_idx"] for x in batch]
        neg_kg_entity_idx_lst = [x["neg_kg_entity_idx"] for x in batch]

        data = {
            "input_ids": pack_tensor_2D(context_ids_lst, default=0, dtype=torch.int64),
            "attention_mask": pack_tensor_2D(context_mask_lst, default=0, dtype=torch.int64),
            "context_idx": context_idx_lst,
            "act_turn_idx": act_turn_idx_lst,
            "entity_turn_idx": new_entity_turn_idx_lst,
            "batch_entity_turn_idx": batch_entity_turn_idx_lst,
            "act_labels": torch.tensor(act_labels_lst, dtype=torch.float32),
            "entity_labels": torch.tensor(entity_labels_lst, dtype=torch.float32),
            "target_kg_entity_idx": target_kg_entity_idx_lst,
            "neg_kg_entity_idx": neg_kg_entity_idx_lst,
        }

        idx = [x["idx"] for x in batch]
        turn_idx = [x["turn_idx"] for x in batch]
        return data, idx, turn_idx
    return collate_function

def construct_data(args, data_type, mode, per_gpu_batch_size, tokenizer, data_sampler):
    batch_size = per_gpu_batch_size * max(1, args.n_gpu)
    dataset = BaseDataset(args.data_name, data_type, mode, tokenizer)
    sampler = data_sampler(dataset)
    collate_fn = get_collate_function(args.data_name)
    dataloader = DataLoader(dataset, sampler=sampler, 
        batch_size=batch_size, num_workers=args.data_num_workers, collate_fn=collate_fn)
    return dataset, dataloader, batch_size
