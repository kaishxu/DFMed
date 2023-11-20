import time
import argparse

def run_parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--mode", type=str, default="train", help="select from [train, test, inference]")
    parser.add_argument("--output_dir", type=str, default="./train")
    parser.add_argument("--base_model_path", type=str, default="../bart-base-chinese")
    parser.add_argument("--data_name", type=str, default="kamed", help="select from [kamed, meddg]")
    parser.add_argument("--data_type", type=str, default="train", help="select from [train, test, valid]")
    parser.add_argument("--train_name", type=str, default=None, help="for recording the objective of this training")
    parser.add_argument("--for_meddg_160", action='store_true', help="use predicted entities on MedDG dataset")
    parser.add_argument("--k_entity", type=int, default=None, help="the retrieved top-k entities for response generation")
    parser.add_argument("--act_entity_dir", type=str, default=None, help="the directory of the predicted acts and entities")
    parser.add_argument("--result_save_dir", type=str, default=None, help="the directory of the generated results")

    ## General parameters
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--decode_max_length", type=int, default=150)
    parser.add_argument("--eval_model_path", type=str, default=None)
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--evaluate_during_training", action="store_true")

    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--data_num_workers", default=0, type=int)

    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int)

    args = parser.parse_args()

    if args.train_name:
        args.log_dir = f"{args.output_dir}/log/{args.train_name}"
        args.model_save_dir = f"{args.output_dir}/models/{args.train_name}"
    else:
        time_stamp = time.strftime("%b-%d_%H:%M:%S", time.localtime())
        args.log_dir = f"{args.output_dir}/log/{time_stamp}"
        args.model_save_dir = f"{args.output_dir}/models/{time_stamp}"
    return args
