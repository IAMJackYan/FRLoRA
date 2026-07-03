from opt import *
import os
import time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
from sklearn.metrics import accuracy_score, f1_score
import random
from dataloaders import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
from dataloaders import build_loaders, build_split_loaders
from trainer import build_local_trainers
import copy
from fed_global import global_aggregate, statistic_FNorm, statistic_Difference, ours_init
import numpy as np


def statistic_lora_f(global_dict):
    global_fnorm = []
    for key in global_dict.keys():
        temp_norm = np.linalg.norm(global_dict[key].cpu().numpy())
        global_fnorm.append(temp_norm)
    global_fnorm = np.array(global_fnorm) 
    return global_fnorm

def evaluate(model, test_loader, device, metric):
    model.eval()
    for step, batch in enumerate(tqdm(test_loader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metric = metric.compute()
    return eval_metric

def nnewsgroup_evaluate(model, test_loader, device):
    model.eval()

    preds = []
    labels = []

    for step, batch in enumerate(tqdm(test_loader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]

        preds.append(predictions)
        labels.append(references)
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro')

    eval_metric = {'accuracy': acc, 'f1': f1}

    return eval_metric
    

args = get_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

peft_config = LoraConfig(
        r=args.peft_lora_r,
        lora_alpha=args.peft_lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )

# setup
#dataset_names = ['qnli', 'qqp', 'sst2', 'rte', 'mrpc']
dataset_names = ['qnli', 'qqp', 'sst2']
num_labels = 2
if args.dataset == '20ng':
    num_labels = 20
elif args.dataset == 'mnli':
    num_labels=3

client_nums = args.num_clients
device = torch.device(args.device)


# load dataloaders
train_loaders, test_loader, sample_numlist, = build_split_loaders(args)

# load models
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path,num_labels=num_labels,return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model = model.to(device)


initial_dict = copy.deepcopy(get_peft_model_state_dict(model))

initial_dict, model = ours_init(args, initial_dict, model)

init_base = copy.deepcopy(initial_dict)

set_peft_model_state_dict(model, initial_dict)
scale = args.peft_lora_alpha / args.peft_lora_r
trainers = build_local_trainers(args.alg, model, args.lr_head, args.lr_lora, train_loaders, args.weight_decay, args.num_epochs, device, args.mu, scale, args.peft_lora_r, args.local_update_step)

training_loss = [[] for i in range(client_nums)]
local_fnorms_record = []
global_fnorms_record = []
acc_list = []
agg_dif = []
lr_difs = []

best_round = 0
best_acc = 0

os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)


delta_w_map = {}

i = 0

pissa_init_records=[]

lora_records = []

for epoch in range(args.num_epochs):

    # t = statistic_lora_f(initial_dict)
    # pissa_init_records.append(t)

    local_dict_list = {}
    print('--------- Round: [{}/{}] ----------'.format(epoch, args.num_epochs))
    for client in range(client_nums):
        trainers[client].set_model_parameters(initial_dict)
        loss = trainers[client].train()
        print('client: {}   Loss: {}'.format(client, loss))
        local_dict_list[client] = trainers[client].get_model_parameters()
        training_loss[client].append(loss)

        weight_save_path = os.path.join(args.output_dir, args.exp_name, 'checkpoint_r_{}_c_{}.pth'.format(epoch, client))


    
    initial_dict, _ = global_aggregate(args, copy.deepcopy(initial_dict), local_dict_list,  sample_numlist, client_nums)

    set_peft_model_state_dict(model, initial_dict)

   
    if args.dataset == 'cola':
        metric = load_metric("glue", args.dataset)
        eval_metric = evaluate(model, test_loader, device, metric)
        acc_list.append(eval_metric['matthews_correlation'])
        print(f"Round: {epoch}:", eval_metric, '\033[32m, current_best_corr:\033[0m',max(acc_list))
    elif args.dataset == '20ng':
        eval_metric = nnewsgroup_evaluate(model, test_loader, device)
        acc_list.append(eval_metric['accuracy'])
        print(f"Round: {epoch}:", eval_metric, '\033[32m, current_best_acc:\033[0m',max(acc_list))
    else:
        metric = load_metric("glue", args.dataset)
        eval_metric = evaluate(model, test_loader, device, metric)
        acc_list.append(eval_metric['accuracy'])
        print(f"Round: {epoch}:", eval_metric, '\033[32m, current_best_acc:\033[0m',max(acc_list))
    
    if eval_metric['accuracy'] > best_acc:
        best_acc = eval_metric['accuracy']
        best_round = epoch
    

    # # merge_weight
    param_names = []
    delta_param_dicts = {}
    scale = args.peft_lora_alpha / args.peft_lora_r
    for key in initial_dict.keys():
        if 'lora_A' in key:
            param_names.append(key.replace('.lora_A.weight', ''))
        elif 'lora_B' in key:
            param_names.append(key.replace('.lora_B.weight', ''))
    param_names = set(param_names)

    global_fnorm = []
    for key in param_names:
        delta_w = initial_dict[key+'.lora_B.weight'] @ initial_dict[key+'.lora_A.weight'] * scale - init_base[key+'.lora_B.weight'] @ init_base[key+'.lora_A.weight']* scale
        temp = model.state_dict()[key+'.weight'] + delta_w 
        model.state_dict()[key+'.weight'].data.copy_(temp)
        for client in range(client_nums):
            trainers[client].model.state_dict()[key+'.weight'].data.copy_(temp)
        
        if epoch==0:
            delta_w_map[key] = delta_w
        else:
            delta_w_map[key] += delta_w
        
        temp_norm = np.linalg.norm(delta_w_map[key].cpu().numpy())
        global_fnorm.append(temp_norm)
    global_fnorm = np.array(global_fnorm) 
    global_fnorms_record.append(global_fnorm)

    for key in initial_dict.keys():
        if "classifier" in key:
            continue
        initial_dict[key] = init_base[key]



    np.save(os.path.join(args.output_dir, args.exp_name, "training_loss.npy"), np.array(training_loss))
    np.save(os.path.join(args.output_dir, args.exp_name, "test_acc.npy"), np.array(acc_list))
    print(best_round, best_acc)

