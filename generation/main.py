import sys
import torch
import logging
from parsing import run_parse_args
from transformers import BertTokenizer, BartForConditionalGeneration
from torch.utils.data import RandomSampler, SequentialSampler

from model import Generator
from dataset import construct_data
from utils import set_seed
from training import train
from evaluating import evaluate
from inference import inference

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

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

    # Model
    if args.mode == "train":
        model_path = args.base_model_path
    else:
        model_path = args.eval_model_path
    model = Generator.from_pretrained(model_path)
    model.to(args.device)

    logger.info("Training/Evaluation parameters %s", args)

    # Train/Evaluate
    if args.mode == "train":
        train_dataset, train_dataloader, args.train_batch_size = construct_data(args, args.data_type, "train", args.per_gpu_train_batch_size, tokenizer, RandomSampler)
        val_dataset, val_dataloader, args.eval_batch_size = construct_data(args, "valid", "evaluate", args.per_gpu_eval_batch_size, tokenizer, SequentialSampler)
        test_dataset, test_dataloader, args.eval_batch_size = construct_data(args, "test", "evaluate", args.per_gpu_eval_batch_size, tokenizer, SequentialSampler)
        train(args, model, train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader)
    else:
        prefix = args.eval_model_path.split("/")[-1]
        eval_dataset, eval_dataloader, args.eval_batch_size = construct_data(args, args.data_type, args.mode, args.per_gpu_eval_batch_size, tokenizer, SequentialSampler)
        if args.mode != "inference":
            result = evaluate(args, model, eval_dataset, eval_dataloader, prefix)
            print('Acc: {}'.format(result[1]))
        else:
            bleu_1, bleu_4, f1, rouge_1, rouge_2 = inference(args, model, eval_dataset, eval_dataloader, prefix)
            print("BLEU-1: {:.2f}".format(bleu_1 * 100), end="\t")
            print("BLEU-4: {:.2f}".format(bleu_4 * 100))
            print("ROUGE-1: {:.2f}".format(rouge_1 * 100), end="\t")
            print("ROUGE-2: {:.2f}".format(rouge_2 * 100))
            print("F1: {:.2f}".format(f1 * 100))

if __name__ == "__main__":
    main()
