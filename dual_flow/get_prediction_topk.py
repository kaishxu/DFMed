import os
import sys
import json
import pickle
import numpy as np
import argparse
import math
from typing import List

act_id = {"INQUIRE": 0, "DIAGNOSIS": 1, "TREATMENT": 2, "TEST": 3, "PRECAUTION": 4, "INFORM": 5, "CHITCHAT": 6}
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

def get_acts(args):

    for _, checkpoints, _ in os.walk(args.saved_model_dir):
        break

    # origin threshold
    tshd = [0.4, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5]

    # rank top-k metrics in valid set
    metric_dict = dict()
    for ckpt in checkpoints:
        with open(f"{args.saved_model_dir}/{ckpt}/{args.data_name}_valid_predicted_act.pk", "rb") as infile:
            test_scores, test_labels, test_context_idx_lst, test_predicted_dict = pickle.load(infile)

        for i in range(7):
            f1, acc, r, p = calculate_metrics(test_scores[:,i], test_labels[:,i], threshold=tshd[i])
            if i not in metric_dict:
                metric_dict[i] = []
            metric_dict[i].append((ckpt, f1))

    for i in range(7):
        metric_dict[i] = sorted(metric_dict[i], key=lambda x: x[1], reverse=True)
        for topk in range(args.top_k):
            with open(f"{args.saved_model_dir}/{metric_dict[i][topk][0]}/{args.data_name}_{args.data_type}_predicted_act.pk", "rb") as infile:
                test_scores, test_labels, test_context_idx_lst, test_predicted_dict = pickle.load(infile)
            if not 'best_scores' in locals().keys():
                best_scores = np.zeros_like(test_scores)
            best_scores[:, i] = np.maximum(best_scores[:, i], test_scores[:, i])

    # modify threshold since the maximum may raise the overall recall rate
    tshd = [0.4, 0.2, 0.25, 0.2, 0.25, 0.45, 0.5]

    tshd = np.ones_like(best_scores) * np.array(tshd)
    pred_labels = best_scores > tshd
    predicted_dict = dict()
    for labels, context_idx in zip(pred_labels, test_context_idx_lst):
        predicted_dict[context_idx] = []
        for i in range(7):
            if labels[i]:
                predicted_dict[context_idx].append(act_dict[i])

    with open(f"{args.saved_result_dir}/{args.data_name}_{args.data_type}_predicted_act.pk", "wb") as outfile:
        pickle.dump((best_scores, test_labels, test_context_idx_lst, predicted_dict), outfile)

def get_ner_entity(data_name):
    ner_entity_lst = []
    ner_entity_dict = {}
    with open(f"../data/{data_name}_ner_entity/{data_name}_entity.txt", "r") as f:
        for i, line in enumerate(f):
            ner_entity_lst.append(line.strip())
            ner_entity_dict[line.strip()] = i
    return ner_entity_lst, ner_entity_dict

def get_ranked_entitis(args, best_metric):

    with open(f"{args.saved_model_dir}/{best_metric['entity']['checkpoint']}/{args.data_name}_{args.data_type}_ranked_entity.pk", "rb") as infile:
        entity_ranked_dict = pickle.load(infile)

    with open(f"{args.saved_result_dir}/{args.data_name}_{args.data_type}_ranked_entity.pk", "wb") as outfile:
        pickle.dump(entity_ranked_dict, outfile)

def get_selected_entitis(args):

    ner_entity_lst, ner_entity_dict = get_ner_entity(args.data_name)
    for _, checkpoints, _ in os.walk(args.saved_model_dir):
        break

    # origin threshold
    tshd = 0.14

    # rank top-k metrics in valid set
    metric_lst = list()
    for ckpt in checkpoints:
        with open(f"{args.saved_model_dir}/{ckpt}/{args.data_name}_valid_selected_entity.pk", "rb") as infile:
            test_scores, test_labels, test_context_idx_lst, test_predicted_dict = pickle.load(infile)

        f1, acc, r, p = calculate_metrics(test_scores, test_labels, threshold=tshd)
        metric_lst.append((ckpt, f1))

    metric_lst = sorted(metric_lst, key=lambda x: x[1], reverse=True)
    for topk in range(args.top_k):
        with open(f"{args.saved_model_dir}/{metric_lst[topk][0]}/{args.data_name}_{args.data_type}_selected_entity.pk", "rb") as infile:
            test_scores, test_labels, test_context_idx_lst, test_predicted_dict = pickle.load(infile)
        if not 'best_scores' in locals().keys():
            best_scores = np.zeros_like(test_scores)
        best_scores = np.maximum(best_scores, test_scores)

    # modify threshold since the maximum may raise the overall recall rate
    tshd = 0.15

    pred_labels = best_scores > tshd
    predicted_dict = dict()
    for labels, context_idx in zip(pred_labels, test_context_idx_lst):
        predicted_dict[context_idx] = []
        for i in range(160):
            if labels[i]:
                predicted_dict[context_idx].append(ner_entity_lst[i])

    with open(f"{args.saved_result_dir}/{args.data_name}_{args.data_type}_selected_entity.pk", "wb") as outfile:
        pickle.dump((best_scores, test_labels, test_context_idx_lst, predicted_dict), outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="kamed", help="select from [kamed, meddg]")
    parser.add_argument("--data_type", type=str, default="train", help="select from [train, test, valid]")
    parser.add_argument("--log_dir", type=str, default="./train/log/demo")
    parser.add_argument("--saved_model_dir", type=str, default="./train/models")
    parser.add_argument("--saved_result_dir", type=str, default="./train/results")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--get_acts", action='store_true')
    parser.add_argument("--get_ranked_entitis", action='store_true')
    parser.add_argument("--get_selected_entitis", action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.saved_result_dir):
        os.makedirs(args.saved_result_dir)

    if args.get_acts:
        get_acts(args)
    if args.get_ranked_entitis:
        # load best metric based on valid set
        with open(f"{args.log_dir}/{args.data_name}_valid_best_metric.json", "r") as infile:
            best_metric = json.load(infile)
        get_ranked_entitis(args, best_metric)
    if args.get_selected_entitis:
        get_selected_entitis(args)
