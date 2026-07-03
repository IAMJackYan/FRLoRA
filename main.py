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
from dataloaders import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
from dataloaders import build_loaders
from trainer import build_local_trainers
import copy
from fed_global import global_aggregate, statistic_FNorm
import numpy as np

def evaluate(model, test_loader, device, metric):
    model.eval()
    for step, batch in enumerate(tqdm(test_loader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        # print(outputs.logits)
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metric = metric.compute()
    return eval_metric
    


args = get_args()

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
client_nums = len(dataset_names)
device = torch.device(args.device)


# load dataloaders
train_loaders, test_loaders, sample_numlist, = build_loaders(dataset_names, args)

# load models
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path,num_labels=num_labels,return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model = model.to(device)

global_dict = copy.deepcopy(get_peft_model_state_dict(model))

trainers = build_local_trainers(args.alg, model, args.lr_head, args.lr_lora, train_loaders, args.weight_decay, args.num_epochs, device)

training_loss = [[] for i in range(client_nums)]
local_fnorms_record = []
global_fnorms_record = []
acc_list = [[] for i in range(client_nums)]

best_round = 0
best_acc = 0

os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)

for epoch in range(args.num_epochs):
    local_dict_list = {}
    print('--------- Round: [{}/{}] ----------'.format(epoch, args.num_epochs))
    for client in range(client_nums):
        trainers[client].set_model_parameters(global_dict)
        loss = trainers[client].train()
        print('client: {}   Loss: {}'.format(dataset_names[client], loss))
        local_dict_list[client] = trainers[client].get_model_parameters()
        training_loss[client].append(training_loss)
    
    global_dict = global_aggregate(args, global_dict, local_dict_list,  sample_numlist, client_nums)

    global_fnorms, local_fnorms = statistic_FNorm(global_dict, local_dict_list, client_nums)
    global_fnorms_record.append(global_fnorms)
    local_fnorms_record.append(local_fnorms)

    set_peft_model_state_dict(model, global_dict)

    for client in range(client_nums):
        metric = load_metric("glue", dataset_names[client])
        eval_metric = evaluate(model, test_loaders[client], device, metric)

        if dataset_names[client] == 'cola':
            acc_list[client].append(eval_metric['matthews_correlation'])
            print(f"client: {dataset_names[client]}:", eval_metric, '\033[32m, current_best_corr:\033[0m',max(acc_list[client]))
        else:
            acc_list[client].append(eval_metric['accuracy'])
            print(f"client: {dataset_names[client]}:", eval_metric, '\033[32m, current_best_acc:\033[0m',max(acc_list[client]))
        
    
    avg_acc = 0

    for i in range(client_nums):
        avg_acc += acc_list[client][epoch]
    
    if avg_acc > best_acc:
        best_acc = avg_acc
        best_round = epoch

    
    # np.save(os.path.join(args.output_dir, args.exp_name, "training_loss.npy"), np.array(training_loss))
    # np.save(os.path.join(args.output_dir, args.exp_name, "global_fnorms.npy"), np.array(global_fnorms_record))
    # np.save(os.path.join(args.output_dir, args.exp_name, "local_fnorms.npy"), np.array(local_fnorms_record))
    # np.save(os.path.join(args.output_dir, args.exp_name, "test_acc.npy"), np.array(acc_list))

print('best_round {} best_acc {}'.format(best_round, best_acc))
for i in range(client_nums):
    print('client: {} acc {}'.format(dataset_names[client], acc_list[client][best_round]))



