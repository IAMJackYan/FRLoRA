import argparse

print('Parsing args')

parser = argparse.ArgumentParser() 
parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
# parser.add_argument("--model_name_or_path", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--n_frequency", type=int, default=200)
parser.add_argument("--lr_head", type=float, default=2e-5)
parser.add_argument("--lr_lora", type=float, default=2e-5)
parser.add_argument("--peft_lora_r", type=int, default=16)
parser.add_argument("--peft_lora_alpha", type=int, default=32)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warm_step", type=float, default=0.06)
parser.add_argument("--train_ratio", type=float, default=1)
parser.add_argument("--scale", type=float, default=100.)
parser.add_argument("--width", type=float, default=200.)
parser.add_argument("--fc", type=float, default=1.)
parser.add_argument("--share_entry", action= "store_true")
parser.add_argument("--set_bias", action= "store_true")
parser.add_argument("--seed", type=int, default=00000)
parser.add_argument("--entry_seed", type=int, default=2024)

parser.add_argument("--mu", type=float, default=0.01)
parser.add_argument("--mom", type=float, default=0.0001)

parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--alg", type=str, default='fedavg')
parser.add_argument("--exp_name", type=str, default='')
parser.add_argument("--output_dir", type=str, default='./output5')
parser.add_argument("--dataset", type=str, default='rte')
parser.add_argument("--num_clients", type=int, default=6)

parser.add_argument("--split_strategy", type=str, default='lda')
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--use_dora_init", type=int, default=0)
parser.add_argument("--local_update_step", type=int, default=10)

parser.add_argument("--fedopt_beta1", type=float, default=0.9)
parser.add_argument("--fedopt_beta2", type=float, default=0.99)

parser.add_argument("--fedopt_tau", type=float, default=1e-3)
parser.add_argument("--fedopt_eta", type=float, default=1e-3)

args = parser.parse_args()

def get_args():
    return parser.parse_args()