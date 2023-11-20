import os
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

def save_model(model, embeds, output_dir, save_name, args):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))

    embeds_np = embeds.cpu().numpy()
    entity_embed_dir = os.path.join(save_dir, f"{args.data_name}_entity_embeds.pk")
    with open(entity_embed_dir, "wb") as outfile:
        pickle.dump(embeds_np, outfile)

def load_entity_embed(input_dir, data_name):
    path = os.path.join(input_dir, f"{data_name}_entity_embeds.pk")
    with open(path, "rb") as infile:
        embeds = pickle.load(infile)
    return torch.from_numpy(embeds)

def squeeze_lst(lst):
    tmp = []
    for x in lst:
        if x not in tmp:
            tmp.append(x)
    return tmp

def get_cmekg_link_specific(data_name):
    head_lst = set()
    with open(f"../data/cmekg/head_{data_name}.txt", "r") as f:
        for line in f:
            head_lst.add(line.strip())

    del_entity_lst = []  #额外单独删除

    cmekg_link_relation_dict = defaultdict(list)
    entity_lst = set()
    cmekg_link_dict = defaultdict(list)
    cmekg_link_set = set()
    useless_r = ["中心词", "ICD-10", "UMLS", "拼音"]
    with open("../data/cmekg/kg_disease.txt", "r") as f:
        for line in f:
            h, r, t = line.split("\t")
            t = t.strip()
            if r == "英文名称" and not t.isupper():  #仅保留英文缩写
                continue
            useless_disease_r = ["进入路径标准", "多发地区", "多发季节", "多发群体", "易感人群", "标准住院时间", "出院标准", "发病机制", "发病率", "发病性别倾向", "发病年龄"]
            if h != t and r not in useless_r and r not in useless_disease_r and h in head_lst and t not in del_entity_lst:
                cmekg_link_set.add(h + "\t" + t)
                cmekg_link_set.add(t + "\t" + h)
                cmekg_link_dict[h].append(t)
                cmekg_link_dict[t].append(h)
                cmekg_link_relation_dict[f"{h}\t{t}"].append(r)
                entity_lst.add(h)
                entity_lst.add(t)

    with open("../data/cmekg/kg_symptom.txt", "r") as f:
        for line in f:
            h, r, t = line.split("\t")
            t = t.strip()
            if r == "英文名称" and not t.isupper():  #仅保留英文缩写
                continue
            useless_symptom_r = ["进入路径标准", "多发地区", "多发季节", "多发群体", "易感人群", "标准住院时间", "出院标准", "发病机制", "发病率", "发病性别倾向", "发病年龄"]
            if h != t and r not in useless_r and r not in useless_symptom_r and h in head_lst and t not in del_entity_lst:
                cmekg_link_set.add(h + "\t" + t)
                cmekg_link_set.add(t + "\t" + h)
                cmekg_link_dict[h].append(t)
                cmekg_link_dict[t].append(h)
                cmekg_link_relation_dict[f"{h}\t{t}"].append(r)
                entity_lst.add(h)
                entity_lst.add(t)

    with open("../data/cmekg/kg_test.txt", "r") as f:
        for line in f:
            h, r, t = line.split("\t")
            t = t.strip()
            if r == "英文名称" and not t.isupper():  #仅保留英文缩写
                continue
            useless_test_r = ["试剂", "原理", "所属分类", "操作方法", "临床意义", "正常值"]
            if h != t and r not in useless_r and r not in useless_test_r and h in head_lst and t not in del_entity_lst:
                cmekg_link_set.add(h + "\t" + t)
                cmekg_link_set.add(t + "\t" + h)
                cmekg_link_dict[h].append(t)
                cmekg_link_dict[t].append(h)
                cmekg_link_relation_dict[f"{h}\t{t}"].append(r)
                entity_lst.add(h)
                entity_lst.add(t)

    with open("../data/cmekg/kg_medicine.txt", "r") as f:
        for line in f:
            h, r, t = line.split("\t")
            t = t.strip()
            useless_medicine_r = ["英文名称", "拉丁学名", "OTC类型", "出处", "分子量",
                                "晶系", "化学式", "比重", "硬度", "采集加工", "执行标准",
                                "批准文号", "有效期", "分布区域", "采收时间", "是否纳入医保",
                                "是否处方药", "药品监管分级", 
                                "贮藏", "界", "门", "纲", "目", "科", "属", "种",
                                "入药部位", "性味", "性状", "特殊药品", "规格", "剂型", "成份", "组成"]
            if h != t and r not in useless_r and r not in useless_medicine_r and h in head_lst and t not in del_entity_lst:
                cmekg_link_set.add(h + "\t" + t)
                cmekg_link_set.add(t + "\t" + h)
                cmekg_link_dict[h].append(t)
                cmekg_link_dict[t].append(h)
                cmekg_link_relation_dict[f"{h}\t{t}"].append(r)
                entity_lst.add(h)
                entity_lst.add(t)

    cmekg_link_dict = dict(cmekg_link_dict)
    cmekg_link_relation_dict = dict(cmekg_link_relation_dict)
    for key in cmekg_link_dict:
        cmekg_link_dict[key] = squeeze_lst(cmekg_link_dict[key])
    return cmekg_link_dict, cmekg_link_set, entity_lst, head_lst

def get_cmekg_entity_specific(data_name):
    entity_lst = []
    entity_dict = dict()
    with open(f"../data/cmekg/entities_{data_name}.txt", "r") as f:
        for i, line in enumerate(f):
            entity_lst.append(line.strip())
            entity_dict[line.strip()] = i
    entity_type_dict = dict()
    for key in ["disease", "medicine", "symptom", "test"]:
        with open(f"../data/cmekg/{key}_lst.txt", "r") as f:
            for line in f:
                entity_type_dict[line.strip()] = key
    return entity_lst, entity_dict, entity_type_dict

def get_ner_entity(data_name):
    ner_entity_lst = []
    ner_entity_dict = {}
    with open(f"../data/{data_name}_ner_entity/{data_name}_entity.txt", "r") as f:
        for i, line in enumerate(f):
            ner_entity_lst.append(line.strip())
            ner_entity_dict[line.strip()] = i
    return ner_entity_lst, ner_entity_dict

def get_entity_matrix(data_name, entity_lst):
    cmekg_link_dict, _, _, head_lst = get_cmekg_link_specific(data_name)
    entity_matrix = torch.eye(len(entity_lst))
    print(f"Num of {data_name} entities in adj matrix: {len(entity_lst)}", )

    for key in tqdm(cmekg_link_dict, desc="Get entity matrix"):
        for entity in cmekg_link_dict[key]:
            entity_matrix[entity_lst.index(key), entity_lst.index(entity)] = 1
    entity_matrix = (1.0 - entity_matrix) * torch.finfo(torch.float32).min

    return entity_matrix.to(torch.device("cpu"))
