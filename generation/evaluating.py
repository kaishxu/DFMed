import sys
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

def evaluate(args, model, eval_dataset, eval_dataloader, prefix):
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval
    logger.info("***** Running Evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    test_loss = 0
    test_corrects = 0
    test_num_targets = 0
    test_idx = []
    test_turn_idx = []

    for batch, idx, turn_idx in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            batch = {k:v.to(args.device) for k, v in batch.items()}
            model.eval()
            outputs = model(**batch)

            loss, logits = outputs.loss, outputs.logits
            test_loss += loss
            test_idx += idx
            test_turn_idx += turn_idx

            shift_logits = logits.contiguous()
            shift_labels = batch["labels"].contiguous()

            _, preds = shift_logits.max(dim=-1)
            not_ignore = shift_labels.ne(-100)
            num_targets = not_ignore.long().sum().item()
            corrects = (shift_labels == preds) & not_ignore
            corrects = corrects.float().sum()

            test_corrects += corrects
            test_num_targets += num_targets

    return test_loss/len(eval_dataloader), test_corrects/test_num_targets
