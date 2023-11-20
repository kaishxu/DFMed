import os
import torch
import random
import numpy as np

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

def save_model(model, output_dir, save_name, args):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))

def squeeze_lst(lst):
    tmp = []
    for x in lst:
        if x not in tmp:
            tmp.append(x)
    return tmp

def get_cmekg_entity_specific(data_name):
    entity_lst = []
    entity_dict = dict()
    with open(f"../data/cmekg/entities_{data_name}.txt", "r") as f:
        for i, line in enumerate(f):
            entity_lst.append(line.strip())
            entity_dict[line.strip()] = i
    return entity_lst, entity_dict
