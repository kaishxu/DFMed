import sys
import pickle
import torch
import torch.nn as nn
import logging
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import SequentialSampler

from evaluating import evaluate
from utils import save_model, set_seed, get_entity_matrix
from dataset_entity import construct_data
from get_entity_embed import get_entity_embeds

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

act_dict = {0: "INQUIRE", 1: "DIAGNOSIS", 2: "TREATMENT", 3: "TEST", 4: "PRECAUTION", 5: "INFORM", 6: "CHITCHAT"}

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

def run_eval(args, model, embeds, entity_matrix, dataset, dataloader, tb_writer, global_step, prefix, data_type):
    (test_loss, test_act_loss, test_entity_loss,
    act_f1, act_accuray, act_recall, act_precision, ent_recall) = evaluate(args, model, embeds, entity_matrix, dataset, dataloader, prefix)
    tb_writer.add_scalar(f'{data_type}/loss', test_loss, global_step)
    tb_writer.add_scalar(f'{data_type}/act_loss', test_act_loss, global_step)
    tb_writer.add_scalar(f'{data_type}/entity_loss', test_entity_loss, global_step)
    for i in range(7):
        tb_writer.add_scalar(f'{data_type}/{act_dict[i]}_f1', act_f1[i], global_step)
        tb_writer.add_scalar(f'{data_type}/{act_dict[i]}_accuray', act_accuray[i], global_step)
        tb_writer.add_scalar(f'{data_type}/{act_dict[i]}_recall', act_recall[i], global_step)
        tb_writer.add_scalar(f'{data_type}/{act_dict[i]}_precision', act_precision[i], global_step)
    tb_writer.add_scalar(f'{data_type}/entity_recall', ent_recall, global_step)

def train(args, model, embeds, train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader, tokenizer):
    # Train the model
    tb_writer = SummaryWriter(args.log_dir)

    # Total steps
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Get entity matrix
    entity_matrix = get_entity_matrix(args.data_name, train_dataset.entity_lst)

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
    _, entity_dataloader, _ = construct_data(args, args.per_gpu_eval_batch_size, tokenizer, SequentialSampler)
    for epoch_idx in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, (batch, idx, turn_idx) in enumerate(epoch_iterator):

            # update entity embed each 100 steps and then fix it
            # gpu memory is not enough to update entity embed each step
            if step % 100 == 0 and global_step < args.entity_update_steps:
                embeds = get_entity_embeds(args, entity_dataloader, model)
            torch.cuda.empty_cache()

            context_idx = batch["context_idx"]
            act_turn_idx = batch["act_turn_idx"]
            entity_turn_idx = batch["entity_turn_idx"]
            batch_entity_turn_idx = batch["batch_entity_turn_idx"]
            target_kg_entity_idx = batch["target_kg_entity_idx"]
            neg_kg_entity_idx = batch["neg_kg_entity_idx"]
            batch = {k:v.to(args.device) for k, v in batch.items() if "idx" not in k and "entity_ids" not in k and "entity_mask" not in k}

            if not args.for_meddg_160:
                batch["entity_labels"] = None

            model.train()
            outputs = model(**batch,
                        entity_embeds=embeds.detach(),
                        entity_matrix=entity_matrix.detach(),
                        context_idx=context_idx,
                        act_turn_idx=act_turn_idx,
                        entity_turn_idx=entity_turn_idx,
                        batch_entity_turn_idx=batch_entity_turn_idx,
                        target_kg_entity_idx=target_kg_entity_idx,
                        neg_kg_entity_idx=neg_kg_entity_idx,
                        )

            loss = outputs[0]
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
                    tb_writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("train/loss", cur_loss, global_step)
                    tb_writer.add_scalar("train/act_loss", outputs[1], global_step)
                    tb_writer.add_scalar("train/entity_loss", outputs[2], global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if global_step % args.eval_steps == 0:
                    save_model(model, embeds, args.model_save_dir, "step-{}".format(global_step), args)

                if args.evaluate_during_training and global_step % args.eval_steps == 0:
                    run_eval(args, model, embeds, entity_matrix, val_dataset, val_dataloader, tb_writer, global_step, prefix="step-{}".format(global_step), data_type="valid")
                    run_eval(args, model, embeds, entity_matrix, test_dataset, test_dataloader, tb_writer, global_step, prefix="step-{}".format(global_step), data_type="test")

        # Save model checkpoint
        save_model(model, embeds, args.model_save_dir, "epoch-{}".format(epoch_idx+1), args)

        if args.evaluate_during_training:
            run_eval(args, model, embeds, entity_matrix, val_dataset, val_dataloader, tb_writer, global_step, prefix="epoch-{}".format(epoch_idx+1), data_type="valid")
            run_eval(args, model, embeds, entity_matrix, test_dataset, test_dataloader, tb_writer, global_step, prefix="epoch-{}".format(epoch_idx+1), data_type="test")
