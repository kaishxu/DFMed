import re
import sys
import pickle
import argparse
from typing import Iterable
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from collections import Counter
import numpy as np
sys.setrecursionlimit(512 * 512 + 10)

def get_youyin(sentence):
    aa = "(吃早饭|吃饭规律|三餐规律|饮食规律|作息规律|辣|油炸|熬夜|豆类|油腻|生冷|煎炸|浓茶|喝酒|抽烟|吃的多|暴饮暴食|不容易消化的食物|情绪稳定|精神紧张|夜宵).*?(吗|啊？|呢？|么|嘛？)"
    bb = "吃得过多过饱|(有没有吃|最近吃|喜欢吃|经常进食|经常吃).*?(辣|油炸|油腻|生冷|煎炸|豆类|不容易消化的食物|夜宵)"
    cc = "(工作压力|精神压力|压力).*?(大不大|大吗)|(心情|情绪|精神).*(怎么样|怎样|如何)|(活动量|运动|锻炼).*?(大|多|少|多不多|咋样|怎么样|怎样).*?(吗|呢)"
    if re.search(aa,sentence) or re.search(bb,sentence) or re.search(cc,sentence):
        return True
    return False

def get_location(sentence):
    cc = "哪个部位|哪个位置|哪里痛|什么部位|什么位置|哪个部位|哪个位置|哪一块|那个部位痛|肚脐眼以上|描述一下位置|具体部位|具体位置"
    if re.search(cc,sentence) is not None:
        return True
    return False

def get_xingzhi(sentence):
    cc = "是哪种疼|怎么样的疼|绞痛|钝痛|隐痛|胀痛|隐隐作痛|疼痛.*性质|(性质|什么样).*(的|得)(痛|疼)"
    if re.search(cc,sentence) is not None:
        return True
    return False

def get_fan(sentence):
    cc = "(饭.*?前|吃东西前|餐前).*(疼|痛|不舒服|不适)|(饭.*?后|吃东西后|餐后).*(疼|痛|不舒服|不适)|(早上|早晨|夜里|半夜|晚饭).*(疼|痛|不舒服|不适)"
    if re.search(cc,sentence) is not None:
        aa = re.search(cc,sentence).span()
        if aa[1] - aa[0] <20:
            return True
    return False

def get_tong_pinglv(sentence):
    cc = "持续的疼|疼一会儿会自行缓解|持续的，还是阵发|症状减轻了没|(疼|痛).*轻|现在没有症状了吗|现在还有症状吗|(一阵一阵|一直|持续).*(疼|痛)|一阵阵.*(痛|疼)|阵发性|持续性"
    if re.search(cc,sentence) is not None:
        aa = re.search(cc,sentence).span()
        return True
    return False

def get_tong(sentence):
    if get_tong_pinglv(sentence) or get_fan(sentence) or get_xingzhi(sentence):
        return True
    return False


def get_other_sym(sentence):
    cc = "(还有什么|还有啥|有没有其|都有啥|都有什么|还有别的|有其他|有没有什么|还有其他).*(症状|不舒服)|别的不舒服|有其他不舒服|主要是什么症状|主要.*症状|哪些不适症状|哪些.*症状|出现了什么症状"
    if re.search(cc,sentence) is not None:
        aa = re.search(cc,sentence).span()
        return True
    return False

def get_time(sentence):
    aa = "(情况|症状|痛|发病|病|感觉|疼|这样|不舒服|大约).*?(多久|多长时间|几周了？|几天了？)"
    bb = "(，|。|、|？)(多长时间了|多久了|有多久了|有多长时间了)|^(多久了|多长时间了|有多久了|有多长时间了|几天了|几周了)"
    cc = "有多长时间|有多久"
    if re.search(aa,sentence) is not None or re.search(bb,sentence) is not None:
        return True
    return False

class KD_Metric():
    def __init__(self) -> None:
        self._pred_true = 0
        self._total_pred = 0
        self._total_true = 0
        with open("../data/new_cy_bii.pk", "rb") as f:
            self.norm_dict = pickle.load(f)

    def reset(self) -> None:
        self._pred_true = 0
        self._total_pred = 0
        self._total_true = 0

    def get_metric(self, reset: bool = False):
        rec, acc, f1 = 0., 0., 0.
        if self._total_pred > 0:
            acc = self._pred_true / self._total_pred
        if self._total_true > 0:
            rec = self._pred_true / self._total_true
        if acc > 0 and rec > 0:
            f1 = acc * rec * 2 / (acc + rec)
        if reset:
            self.reset()
        return {"rec":rec, "acc":acc, "f1":f1}

    def convert_sen_to_entity_set(self, sen):
        entity_set = set()
        for entity in self.norm_dict.keys():
            if entity in sen:
                entity_set.add(self.norm_dict[entity])
        if get_location(sen):
            entity_set.add("位置")
        if get_youyin(sen):
            entity_set.add("诱因")
        if get_tong(sen):
            entity_set.add("性质")
        if get_time(sen):
            entity_set.add("时长")
        return entity_set

    def __call__(
        self,
        references, # list(list(str))
        hypothesis, # list(list(str))
    ) -> None:
        for batch_num in range(len(references)):
            ref = "".join(references[batch_num])
            hypo = "".join(hypothesis[batch_num])
            hypo_list = self.convert_sen_to_entity_set(hypo)
            ref_list = self.convert_sen_to_entity_set(ref)

            self._total_true += len(ref_list)
            self._total_pred += len(hypo_list)
            for entity in hypo_list:
                if entity in ref_list:
                    self._pred_true += 1

def get_entity_acc(ref_dir, hyp_dir):
    kd_metric = KD_Metric()

    ref = []
    with open(ref_dir, "r") as f:
        for line in f:
            text = line.split()[-1]
            text_lst = [x for x in text.strip()]
            ref.append(text_lst)

    hyp = []
    with open(hyp_dir, "r") as f:
        for line in f:
            text = line.split()[-1]
            text_lst = [x for x in text.strip()]
            hyp.append(text_lst)

    kd_metric(ref, hyp)
    scores = kd_metric.get_metric()
    return scores

class NLTK_BLEU():
    def __init__(
        self,
        ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
    ) -> None:
        self._ngram_weights = ngram_weights
        self._scores = []
        self.smoothfunc = SmoothingFunction().method7

    def reset(self) -> None:
        self._scores = []

    def get_metric(self, reset: bool = False):
        score = 0.
        if len(self._scores):
            score = sum(self._scores) / len(self._scores)
        if reset:
            self.reset()
        return score

    def __call__(
        self,
        references, # list(list(str))
        hypothesis, # list(list(str))
    ) -> None:
        for batch_num in range(len(references)):
            if len(hypothesis[batch_num]) <= 1:
                self._scores.append(0)
            else:
                self._scores.append(sentence_bleu([references[batch_num]], hypothesis[batch_num],
                                            smoothing_function=self.smoothfunc,
                                            weights=self._ngram_weights))

def get_bleu(ref_dir, hyp_dir):
    bleu1 = NLTK_BLEU(ngram_weights=(1, 0, 0, 0))
    bleu2 = NLTK_BLEU(ngram_weights=(0.5, 0.5, 0, 0))
    bleu4 = NLTK_BLEU(ngram_weights=(0.25, 0.25, 0.25, 0.25))

    ref = []
    with open(ref_dir, "r") as f:
        for line in f:
            text = line.split()[-1]
            text_lst = [x for x in text.strip()]
            ref.append(text_lst)

    hyp = []
    with open(hyp_dir, "r") as f:
        for line in f:
            text = line.split()[-1]
            text_lst = [x for x in text.strip()]
            hyp.append(text_lst)

    print("Num of Samples:",len(ref))

    bleu1(ref, hyp)
    bleu2(ref, hyp)
    bleu4(ref, hyp)
    scores = {
        "bleu-1": bleu1.get_metric(),
        "bleu-2": bleu2.get_metric(),
        "bleu-4": bleu4.get_metric(),
    }
    return scores

def get_rouge(ref_dir, hyp_dir):
    ref = []
    with open(ref_dir, "r") as f:
        for line in f:
            text = line.split()[-1]
            ref.append(text.strip())

    hyp = []
    with open(hyp_dir, "r") as f:
        for line in f:
            text = line.split()[-1]
            hyp.append(text.strip())

    pred_cleaned = []
    target_cleaned = []
    for x, y in zip(hyp, ref):
        if x != "" and y != "":
            pred_cleaned.append(" ".join(x))
            target_cleaned.append(" ".join(y))

    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_cleaned, target_cleaned, avg=True)
    scores = {
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
    }

    return scores

def distinct(hyp_dir):
    """ Calculate intra/inter distinct 1/2. """
    hyp = []
    with open(hyp_dir, "r") as f:
        for line in f:
            text = line.split()[-1]
            hyp.append(text.strip())

    batch_size = len(hyp)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in hyp:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hp", help="generated samples")
    parser.add_argument("--rf", help="reference samples")
    args = parser.parse_args()

    scores = get_bleu(args.rf, args.hp)
    scores.update(get_entity_acc(args.rf, args.hp))
    scores.update(get_rouge(args.rf, args.hp))
    print("BLEU-1:", round(scores["bleu-1"]*100, 2), end="\t")
    print("BLEU-2:", round(scores["bleu-2"]*100, 2), end="\t")
    print("BLEU-4:", round(scores["bleu-4"]*100, 2))
    print("Rouge-1:", round(scores["rouge-1"]*100, 2), end="\t")
    print("Rouge-2:", round(scores["rouge-2"]*100, 2), end="\t")
    print("Rouge-L:", round(scores["rouge-l"]*100, 2))
    print("Entity-F1:", round(scores["f1"]*100, 2))
    print("Entity-R:", round(scores["rec"]*100, 2))
    print("Entity-P:", round(scores["acc"]*100, 2))
    print(distinct(args.hp))
