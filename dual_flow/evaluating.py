import os
import sys
import json
import math
import pickle
import torch
import logging
import numpy as np
from tqdm import tqdm

from utils import get_cmekg_entity_specific, get_ner_entity

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

act_dict = {0: "INQUIRE", 1: "DIAGNOSIS", 2: "TREATMENT", 3: "TEST", 4: "PRECAUTION", 5: "INFORM", 6: "CHITCHAT"}

def calculate_metrics(scores, labels, threshold):
    ep = 1e-8
    TP = ((scores > threshold) & (labels == 1)).sum()
    TN = ((scores < threshold) & (labels == 0)).sum()
    FN = ((scores < threshold) & (labels == 1)).sum()
    FP = ((scores > threshold) & (labels == 0)).sum()

    p = TP / (TP + FP + ep)
    r = TP / (TP + FN + ep)
    f1 = 2 * r * p / (r + p + ep)
    acc = (TP + TN) / (TP + TN + FP + FN + ep)
    return f1, acc, r, p

def evaluate(args, model, embeds, entity_matrix, eval_dataset, eval_dataloader, prefix):
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval
    logger.info("***** Running Evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    test_loss = 0
    test_act_loss = 0
    test_entity_loss = 0
    test_act_scores = []
    test_act_labels = []
    test_entity_scores = []
    test_entity_labels = []
    test_idx = []
    test_turn_idx = []

    check_lst = []
    context_embeds = []

    # gpu memory is not enough so GAT is applied on cpu
    # to update all entity embeddings
    embeds_tmp_save_path = f"{args.model_save_dir}/{prefix}/{args.data_name}_{eval_dataset.data_type}_embeds_tmp.pk"
    if os.path.exists(embeds_tmp_save_path):
        with open(embeds_tmp_save_path, "rb") as infile:
            embeds_tmp = pickle.load(infile)
    else:
        embeds_tmp = embeds.to(torch.device("cpu"))
        with torch.no_grad():
            model.entity_gat.to(torch.device("cpu"))
            embeds_tmp, _ = model.entity_gat((embeds_tmp, entity_matrix))
            with open(embeds_tmp_save_path, "wb") as outfile:
                pickle.dump(embeds_tmp, outfile)
    embeds_tmp = embeds_tmp.to(args.device)
    model.entity_gat.to(args.device)

    model.eval()
    for i, (batch, idx, turn_idx) in enumerate(tqdm(eval_dataloader, desc="Evaluating")):

        with torch.no_grad():
            context_idx = batch["context_idx"]
            act_turn_idx = batch["act_turn_idx"]
            entity_turn_idx = batch["entity_turn_idx"]
            batch_entity_turn_idx = batch["batch_entity_turn_idx"]
            target_kg_entity_idx = batch["target_kg_entity_idx"]
            neg_kg_entity_idx = batch["neg_kg_entity_idx"]
            batch = {k:v.to(args.device) for k, v in batch.items() if "idx" not in k and "entity_ids" not in k and "entity_mask" not in k}

            if not args.for_meddg_160:
                batch["entity_labels"] = None

            outputs = model(**batch,
                        entity_embeds=embeds_tmp,
                        context_idx=context_idx,
                        act_turn_idx=act_turn_idx,
                        entity_turn_idx=entity_turn_idx,
                        batch_entity_turn_idx=batch_entity_turn_idx,
                        target_kg_entity_idx=target_kg_entity_idx,
                        neg_kg_entity_idx=neg_kg_entity_idx)

            loss, act_loss, entity_loss, scores_n_labels, context_hiddens = outputs
            act_scores, act_labels, entity_scores, entity_labels = scores_n_labels
            test_loss += loss
            test_act_loss += act_loss
            test_entity_loss += entity_loss
            test_act_scores.append(act_scores)
            test_act_labels.append(act_labels)
            test_entity_scores.append(entity_scores)
            test_entity_labels.append(entity_labels)
            test_idx += idx
            test_turn_idx += turn_idx

            for i, (sample_idx, sample_turn_idx) in enumerate(zip(idx, turn_idx)):
                check_lst.append(sample_idx + "_" + sample_turn_idx)
            context_embeds.append(context_hiddens)

    if os.path.exists(f"{args.log_dir}/{args.data_name}_{eval_dataset.data_type}_best_metric.json"):
        with open(f"{args.log_dir}/{args.data_name}_{eval_dataset.data_type}_best_metric.json", "r") as infile:
            best_metric = json.load(infile)
    else:
        best_metric = {
            "act": {
                "INQUIRE": {"f1": 0, "acc": 0, "recall": 0, "precision": 0, "checkpoint": ""},
                "DIAGNOSIS": {"f1": 0, "acc": 0, "recall": 0, "precision": 0, "checkpoint": ""},
                "TREATMENT": {"f1": 0, "acc": 0, "recall": 0, "precision": 0, "checkpoint": ""},
                "TEST": {"f1": 0, "acc": 0, "recall": 0, "precision": 0, "checkpoint": ""},
                "PRECAUTION": {"f1": 0, "acc": 0, "recall": 0, "precision": 0, "checkpoint": ""},
                "INFORM": {"f1": 0, "acc": 0, "recall": 0, "precision": 0, "checkpoint": ""},
                "CHITCHAT": {"f1": 0, "acc": 0, "recall": 0, "precision": 0, "checkpoint": ""},
            },
            "entity": {"f1": 0, "acc": 0, "recall": 0, "precision": 0, "checkpoint": ""}
        }

    if args.result_save_dir:
        result_save_dir = args.result_save_dir
    else:
        result_save_dir = f"{args.model_save_dir}/{prefix}"

    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    ################
    # Act Evaulate #
    ################

    test_act_scores = torch.cat(test_act_scores, dim=0).cpu().numpy()
    test_act_labels = torch.cat(test_act_labels, dim=0).cpu().numpy()

    tshd = [0.4, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5]
    tshd = np.ones_like(test_act_labels) * np.array(tshd)

    context_idx_lst = []
    for idx, turn_idx in zip(test_idx, test_turn_idx):
        context_idx_lst.append(str(idx) + "_" + str(turn_idx))

    pred_labels = test_act_scores > tshd
    act_predicted_dict = dict()
    for labels, context_idx in zip(pred_labels, context_idx_lst):
        act_predicted_dict[context_idx] = []
        for i in range(7):
            if labels[i]:
                act_predicted_dict[context_idx].append(act_dict[i])
    with open(f"{result_save_dir}/{args.data_name}_{eval_dataset.data_type}_predicted_act.pk", "wb") as outfile:
        pickle.dump((test_act_scores, test_act_labels, context_idx_lst, act_predicted_dict), outfile)

    act_f1 = []  # individual metric for each act
    act_acc = []
    act_r = []
    act_p = []
    for i in range(7):
        f1, acc, r, p = calculate_metrics(test_act_scores[:,i], test_act_labels[:,i], tshd[:,i])
        act_f1.append(f1)
        act_acc.append(acc)
        act_r.append(r)
        act_p.append(p)

        #################
        # Save Best Act #
        #################

        if f1 > best_metric["act"][act_dict[i]]["f1"]:
            best_metric["act"][act_dict[i]]["f1"] = f1
            best_metric["act"][act_dict[i]]["acc"] = acc
            best_metric["act"][act_dict[i]]["recall"] = r
            best_metric["act"][act_dict[i]]["precision"] = p
            best_metric["act"][act_dict[i]]["checkpoint"] = prefix

    ###################
    # Entity Evaulate #
    ###################

    if args.for_meddg_160:
        # For the MedDG dataset
        # ner entity
        ner_entity_lst, _ = get_ner_entity(args.data_name)

        test_entity_scores = torch.cat(test_entity_scores, dim=0).cpu().numpy()
        test_entity_labels = torch.cat(test_entity_labels, dim=0).cpu().numpy()

        tshd = 0.14

        f1, acc, r, p = calculate_metrics(test_entity_scores, test_entity_labels, tshd)
        ent_r = f1

        pred_labels = test_entity_scores > tshd
        entity_predicted_dict = dict()
        for labels, context_idx in zip(pred_labels, context_idx_lst):
            entity_predicted_dict[context_idx] = []
            for i in range(160):
                if labels[i]:
                    entity_predicted_dict[context_idx].append(ner_entity_lst[i])

        with open(f"{result_save_dir}/{args.data_name}_{eval_dataset.data_type}_selected_entity.pk", "wb") as outfile:
            pickle.dump((test_entity_scores, test_entity_labels, context_idx_lst, entity_predicted_dict), outfile)

        ####################
        # Save Best Entity #
        ####################

        if f1 > best_metric["entity"]["f1"]:
            best_metric["entity"]["f1"] = f1
            best_metric["entity"]["acc"] = acc
            best_metric["entity"]["recall"] = r
            best_metric["entity"]["precision"] = p
            best_metric["entity"]["checkpoint"] = prefix

    else:
        # For the KaMed dataset
        # context embeds
        context_embeds = torch.cat(context_embeds, dim=0)

        # entity dict
        _, entity_dict, _ = get_cmekg_entity_specific(args.data_name)

        # 目标kg entity字典(MS MARCO格式)
        context_to_kg_entity = dict()
        with open(f"./{args.data_name}_{eval_dataset.data_type}_reference_entity.txt", "r") as f:
            for line in f:
                line = line.strip().split('\t')
                context_idx = line[0]
                if context_idx in context_to_kg_entity:
                    pass
                else:
                    context_to_kg_entity[context_idx] = set()
                context_to_kg_entity[context_idx].add(entity_dict[line[2]])

        # 目标sub-graph
        with open(f"../data/{args.data_name}_{eval_dataset.data_type}_sub_kg.pk", "rb") as f:
            sub_kg = pickle.load(f)

        ent_r = 0
        total_num = 0
        entity_ranked_dict = dict()
        for i, context_idx in enumerate(check_lst):
            sub_kg_idx = [entity_dict[x] for x in sub_kg[context_idx]]  #sub-graph中的候选entity
            if sub_kg_idx != []:
                sub_kg_entity_embeds = embeds[sub_kg_idx]  #选择相应embed
                score = torch.matmul(context_embeds[i:i+1], sub_kg_entity_embeds.T).squeeze()  #计算相似度
                rank = score.sort(descending=True).indices.tolist()  #获取排序后的候选entity位置标签
                if isinstance(rank, int):  #如果只有一个候选entity
                    rank = [rank]
                entity_ranked = [sub_kg_idx[idx] for idx in rank]  #位置标签对应相应entity idx
                entity_ranked_dict[context_idx] = entity_ranked[:50]  #保存前50的entity

                if context_idx in context_to_kg_entity:  #计算召回率
                    count = 0
                    for entity in entity_ranked[:20]:  #评估前20的entity
                        if entity in context_to_kg_entity[context_idx]:
                            count += 1
                    ent_r += count / len(context_to_kg_entity[context_idx])  #所有target中的召回比例
                    total_num += 1
            else:
                entity_ranked_dict[context_idx] = []
        ent_r = ent_r / total_num

        with open(f"{result_save_dir}/{args.data_name}_{eval_dataset.data_type}_ranked_entity.pk", "wb") as outfile:
            pickle.dump(entity_ranked_dict, outfile)

        ####################
        # Save Best Entity #
        ####################

        if ent_r > best_metric["entity"]["recall"]:
            best_metric["entity"]["recall"] = ent_r
            best_metric["entity"]["checkpoint"] = prefix

    # output best_metric in json file
    with open(f"{args.log_dir}/{args.data_name}_{eval_dataset.data_type}_best_metric.json", "w") as outfile:
        json.dump(best_metric, outfile, indent=4, separators=(',', ': '))

    return (test_loss/len(eval_dataloader), test_act_loss/len(eval_dataloader), test_entity_loss/len(eval_dataloader),
            act_f1, act_acc, act_r, act_p, ent_r)
