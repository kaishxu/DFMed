import sys
import torch
import torch.nn as nn
import logging
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names

from inference import inference
from evaluating import evaluate
from utils import save_model, set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

def get_optimizer(args, model):
    # Parameters with decaying
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [
        name for name in decay_parameters if "bias" not in name
    ]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if n in decay_parameters
            ],
            "lr":
            args.lr,
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if n not in decay_parameters
            ],
            "lr":
            args.lr,
            "weight_decay":
            0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    return optimizer

def run_inf(args, model, dataset, dataloader, tb_writer, global_step, prefix, data_type):
    bleu_1, bleu_4, f1, rouge_1, rouge_2 = inference(args, model, dataset, dataloader, prefix=prefix)
    tb_writer.add_scalar(f'{data_type}/bleu-1', bleu_1, global_step)
    tb_writer.add_scalar(f'{data_type}/bleu-4', bleu_4, global_step)
    tb_writer.add_scalar(f'{data_type}/entity-f1', f1, global_step)
    tb_writer.add_scalar(f'{data_type}/rouge-1', rouge_1, global_step)
    tb_writer.add_scalar(f'{data_type}/rouge-2', rouge_2, global_step)

def train(args, model, train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader):
    # Train the model
    tb_writer = SummaryWriter(args.log_dir)

    # Total steps
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = get_optimizer(args, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running Training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch_idx in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, (batch, idx, turn_idx) in enumerate(epoch_iterator):

            batch = {k:v.to(args.device) for k, v in batch.items()}
            model.train()
            outputs = model(**batch)
            loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                optimizer.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    cur_loss =  (tr_loss - logging_loss)/args.logging_steps
                    tb_writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('train/loss', cur_loss, global_step)
                    logging_loss = tr_loss

                if epoch_idx >= 5 and global_step % 5000 == 0:
                    # Save model checkpoint
                    save_model(model, args.model_save_dir, 'step-{}'.format(global_step), args)

                    run_inf(args, model, val_dataset, val_dataloader, tb_writer, global_step, prefix="step-{}".format(global_step), data_type="valid")

        # Save model checkpoint
        save_model(model, args.model_save_dir, 'epoch-{}'.format(epoch_idx+1), args)

        if args.evaluate_during_training:
            test_loss, test_gen_accuray = evaluate(args, model, val_dataset, val_dataloader, prefix="epoch-{}".format(epoch_idx+1))
            tb_writer.add_scalar('valid/loss', test_loss, epoch_idx+1)
            tb_writer.add_scalar('valid/gen_accuray', test_gen_accuray, epoch_idx+1)

            run_inf(args, model, val_dataset, val_dataloader, tb_writer, global_step, prefix="epoch-{}".format(epoch_idx+1), data_type="valid")
