import os
import sys
import json
import torch
import logging
from tqdm import tqdm
from collections import defaultdict
from transformers import BertTokenizer
from metrics import get_bleu, get_entity_acc, get_rouge

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

def inference(args, model, eval_dataset, eval_dataloader, prefix):
    # multi-gpu inference
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.base_model_path)

    # Inference
    logger.info("***** Running Inference {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    args.early_stopping = True if args.num_beams > 1 else None
    args.do_sample = True if args.top_k != None or args.top_p != 1 else None
    print("Do smaple:", args.do_sample)

    generate_text_dict = defaultdict(dict)
    generate_texts = []
    for batch, idx, turn_idx in tqdm(eval_dataloader, desc="Inference"):
        with torch.no_grad():
            del batch["labels"], batch["decoder_input_ids"], batch["decoder_attention_mask"]  # delete labels
            batch = {k:v.to(args.device) for k, v in batch.items()}
            model.eval()
            outputs = model.generate(**batch,
                                    num_beams=args.num_beams,
                                    early_stopping=args.early_stopping,
                                    do_sample=args.do_sample,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    max_length=args.decode_max_length,
                                    bos_token_id=tokenizer.cls_token_id,
                                    eos_token_id=tokenizer.sep_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    )
            for i, output in enumerate(outputs):
                text = tokenizer.decode(output, skip_special_tokens=True)
                generate_texts.append((idx[i], turn_idx[i], text.replace(" ", "")))
                generate_text_dict[idx[i]][turn_idx[i]] = text.replace(" ", "")

    if args.result_save_dir:
        result_save_dir = args.result_save_dir
    else:
        result_save_dir = f"{args.model_save_dir}/{prefix}"

    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    reference_text_dict = defaultdict(dict)
    ref_file = f"{result_save_dir}/{eval_dataset.data_name}_{eval_dataset.data_type}_reference.txt"
    with open(ref_file, "w") as outfile:
        for (idx, turn_idx, target) in zip(eval_dataset.samples_idx, eval_dataset.samples_turn_idx, eval_dataset.samples_target_raw):
            outfile.write(str(idx) + "\t")
            outfile.write(str(turn_idx) + "\t")
            outfile.write(target + "\n")
            reference_text_dict[idx][turn_idx] = target

    hyp_file = f"{result_save_dir}/{eval_dataset.data_name}_{eval_dataset.data_type}_generate.txt"
    with open(hyp_file, "w") as outfile:
        for text in generate_texts:
            outfile.write(str(text[0]) + "\t")
            outfile.write(str(text[1]) + "\t")
            outfile.write(text[2] + "\n")

    bleu = get_bleu(ref_file, hyp_file)
    acc = get_entity_acc(ref_file, hyp_file)
    rouge = get_rouge(ref_file, hyp_file)

    metric_file = f"{result_save_dir}/{eval_dataset.data_name}_{eval_dataset.data_type}_metric.json"
    with open(metric_file, "w") as outfile:
        json.dump({"bleu-1": bleu["bleu-1"], 
                   "bleu-2": bleu["bleu-2"], 
                   "bleu-4": bleu["bleu-4"], 
                   "entity-f1": acc["f1"], 
                   "rouge-1": rouge["rouge-1"], 
                   "rouge-2": rouge["rouge-2"]},
                  outfile, indent=4, separators=(',', ': '))
    return bleu["bleu-1"], bleu["bleu-4"], acc["f1"], rouge["rouge-1"], rouge["rouge-2"]
