import sys
import pickle
import torch
import logging
from tqdm import tqdm
from parsing import run_parse_args
from transformers import BertTokenizer, BartForConditionalGeneration
from torch.utils.data import RandomSampler, SequentialSampler

from model import ActEntityModel
from dataset import construct_data
from utils import set_seed, load_entity_embed, get_entity_matrix
from training import train
from evaluating import evaluate

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

act_dict = {0: "INQUIRE", 1: "DIAGNOSIS", 2: "TREATMENT", 3: "TEST", 4: "PRECAUTION", 5: "INFORM", 6: "CHITCHAT"}

def output_reference_entity(args, dataset, data_type):
    with open(f"./{args.data_name}_{data_type}_reference_entity.txt", "w") as outfile:
        for sample_raw in dataset.samples_context_raw:
            for key in sample_raw["target_kg_entity"]:
                outfile.write(sample_raw["idx"] + "_" + sample_raw["turn_idx"] + "\t0\t" + str(key[0]) + "\t1\n")

def main():
    args = run_parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.base_model_path)
    
    # Entity Embeddings
    embeds = None

    # Model
    if args.mode == "train":
        model_path = args.base_model_path
    else:
        model_path = args.eval_model_path
        embeds = load_entity_embed(model_path, args.data_name)
        embeds = embeds.to(args.device)
    model = ActEntityModel.from_pretrained(model_path)
    model.act_weight = args.act_weight
    model.entity_weight = args.entity_weight
    model.to(args.device)

    logger.info("Training/Evaluation parameters %s", args)

    # Train/Evaluate
    if args.mode == "train":
        train_dataset, train_dataloader, args.train_batch_size = construct_data(args, args.data_type, "train", args.per_gpu_train_batch_size, tokenizer, RandomSampler)
        val_dataset, val_dataloader, args.eval_batch_size = construct_data(args, "valid", "evaluate", args.per_gpu_eval_batch_size, tokenizer, SequentialSampler)
        test_dataset, test_dataloader, args.eval_batch_size = construct_data(args, "test", "evaluate", args.per_gpu_eval_batch_size, tokenizer, SequentialSampler)
        output_reference_entity(args, val_dataset, "valid")
        output_reference_entity(args, test_dataset, "test")
        train(args, model, embeds, train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader, tokenizer)
    elif args.mode == "evaluate":
        prefix = args.eval_model_path.split("/")[-1]
        test_dataset, test_dataloader, args.eval_batch_size = construct_data(args, args.data_type, "evaluate", args.per_gpu_eval_batch_size, tokenizer, SequentialSampler)
        output_reference_entity(args, test_dataset, args.data_type)
        entity_matrix = get_entity_matrix(args.data_name, test_dataset.entity_lst)
        (_, _, _, act_f1, act_accuray, act_recall, act_precision, ent_recall) = evaluate(args, model, embeds, entity_matrix, test_dataset, test_dataloader, prefix)
        for i in range(7):
            print("{}: F1: {}, Acc: {}, Recall: {} Precision: {}".format(act_dict[i], round(act_f1[i], 4), round(act_accuray[i], 4), round(act_recall[i], 4), round(act_precision[i], 4)))
            print()
        print("{}: Recall: {}".format("Entity", round(ent_recall, 4)))

if __name__ == "__main__":
    main()
