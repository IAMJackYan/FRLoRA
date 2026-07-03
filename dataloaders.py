from opt import *
import os
import time
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
import numpy as np

from transformers import DataCollatorWithPadding

def dirichlet_distribution_noniid_slice(label,
                                        client_num,
                                        alpha,
                                        seed,
                                        min_size=1,
                                        ):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid
    partition/noniid_partition.py

    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """

    np.random.seed(seed)

    if len(label.shape) != 1:
        raise ValueError('Only support single-label tasks!')

    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f'The number of sample should be ' \
                                        f'greater than' \
                                        f' {client_num * min_size}.'
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            # prop = np.array([
            #    p * (len(idx_j) < num / client_num)
            #    for p, idx_j in zip(prop, idx_slice)
            # ])
            # prop = prop / sum(prop)
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]
            size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])
    return idx_slice


def shard_split(label, seed, ratios):
    np.random.seed(seed)
    if len(label.shape) != 1:
        raise ValueError('Only support single-label tasks!')

    num = len(label)
    classes = len(np.unique(label))

    label_indices = {0: np.where(label == 0)[0].tolist(), 1: np.where(label == 1)[0].tolist()}

    num_clients = len(ratios)
    length = min(len(label_indices[0]) // num_clients, len(label_indices[1]) // num_clients)
    
    i = 0
    split_data = []
    for  ratio in ratios:
        part_0_count = int(ratio[0] * length)
        part_1_count = int(ratio[1] * length)
        part_0_indices = label_indices[0][length*i:length*i+part_0_count]
        part_1_indices = label_indices[1][length*i:length*i+part_1_count]
        part_indices = part_0_indices + part_1_indices
        np.random.shuffle(part_indices)
        split_data.append(part_indices)
        i+=1
    
    return split_data

args = get_args()

if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def geneal_tokenize_func(examples):
    outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=args.max_length)
    return outputs

def SST2_tokenize_func(examples):
    outputs = tokenizer(examples["sentence"], truncation=True, max_length=args.max_length)
    return outputs

def QNLI_tokenize_func(examples):
    outputs = tokenizer(examples["question"],examples["sentence"], truncation=True, max_length=args.max_length)
    return outputs

def QQP_tokenize_func(examples):
    outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=args.max_length)
    return outputs

def semeval_tokenize_func(examples):
    outputs = tokenizer(examples["sentence"], padding='max_length', truncation=True, max_length=args.max_length)
    return outputs  

def mnli_tokenize_func(examples):
    outputs = tokenizer(examples["premise"], examples["hypothesis"],  padding='max_length', truncation=True, max_length=args.max_length)
    return outputs 

def newsgroup_tokenize_func(examples):
    outputs = tokenizer(examples["text"], padding='max_length', truncation=True, max_length=args.max_length)
    return outputs

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def build_datasets(task):

    if task == 'sst2' or task == 'cola':
        datasets = load_from_disk(os.path.join('./datasets', task))
        tokenized_datasets = datasets.map(
            SST2_tokenize_func,
            batched=True,
            remove_columns=["idx", "sentence"],
        )
    elif task == 'qnli':
        datasets = load_from_disk(os.path.join('./datasets', task))
        tokenized_datasets = datasets.map(
        QNLI_tokenize_func,
        batched=True,
        remove_columns=["idx", "question", "sentence"],
        )
    elif task == 'qqp':
        datasets = load_from_disk(os.path.join('./datasets', task))
        tokenized_datasets = datasets.map(
        QQP_tokenize_func,
        batched=True,
        remove_columns=["idx", "question1", "question2"],
        )
    elif task == 'mnli':
        datasets = load_from_disk(os.path.join('./datasets', task))
        tokenized_datasets = datasets.map(
        mnli_tokenize_func,
        batched=True,
        remove_columns=["idx", "premise", "hypothesis"],
        )

    elif task == 'semeval':
        datasets = load_dataset("SemEvalWorkshop/sem_eval_2010_task_8")
        tokenized_datasets = datasets.map(
        semeval_tokenize_func,
        batched=True,
        )
        tokenized_datasets=tokenized_datasets.rename_column("relation", "label")
    elif task == '20ng':
        datasets = load_from_disk("datasets/20_newsgroups")
        tokenized_datasets = datasets.map(
        newsgroup_tokenize_func,
        batched=True,
        )
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    elif task == 'imdb':
        datasets = load_dataset("stanfordnlp/imdb", 'train')
        tokenized_datasets = datasets.map(
        newsgroup_tokenize_func,
        batched=True,
        )
    else:
        datasets = load_from_disk(os.path.join('./datasets', task))
        tokenized_datasets = datasets.map(
        geneal_tokenize_func,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
        )

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    return tokenized_datasets 

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


def build_loaders(dataset_names, args):
    train_dataloaders = []
    eval_dataloaders = []
    sample_nums = []
    
    for name in dataset_names:
        
        tokenized_datasets  = build_datasets(name)
        train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=args.bs)
        eval_dataloader = DataLoader(
            tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=args.bs
        )
        train_dataloaders.append(train_dataloader)
        eval_dataloaders.append(eval_dataloader)
        sample_nums.append(len(tokenized_datasets["train"]))

        print('Successfully build dataset {}  length  {}'.format(name, len(tokenized_datasets["train"])))
    
    return train_dataloaders, eval_dataloaders, sample_nums


def split_dataset(args,  dataset):
    dataset = dataset.shuffle(seed=args.seed)        # Shuffle the dataset
    local_datasets = []
    if args.split_strategy == "iid":
        for i in range(args.num_clients):
            local_datasets.append(dataset.shard(args.num_clients, i))
    elif args.split_strategy == "lda":
        labels = dataset['labels']
        idx_slice = dirichlet_distribution_noniid_slice(np.array(labels),
                                        args.num_clients,
                                        args.alpha, args.seed)
        for i in range(args.num_clients):
            local_datasets.append(dataset.select(idx_slice[i]))
    elif args.split_strategy == 'shard':
        labels = dataset['labels']
        idx_slice = shard_split(np.array(labels), args.seed, [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        for i in range(3):
            local_datasets.append(dataset.select(idx_slice[i]))
    
    return local_datasets




def build_split_loaders(args):
    train_dataloaders = []
    sample_nums = []
    
    tokenized_dataset = build_datasets(args.dataset)

    local_datasets = split_dataset(args, tokenized_dataset["train"])

    for i in range(args.num_clients):
        if args.dataset in ['20ng', 'imdb']: 
            train_dataloader = DataLoader(local_datasets[i], shuffle=True, collate_fn=data_collator, batch_size=args.bs)
        else:
            train_dataloader = DataLoader(local_datasets[i], shuffle=True, collate_fn=collate_fn, batch_size=args.bs)
  
        train_dataloaders.append(train_dataloader)
        sample_nums.append(len(local_datasets[i]))
    
    if args.dataset in ['20ng', 'imdb']: 
        eval_dataloader = DataLoader(
                tokenized_dataset["test"], shuffle=False, collate_fn=data_collator, batch_size=args.bs
            )
    elif args.dataset == 'mnli':
        eval_dataloader = DataLoader(
                tokenized_dataset["validation_mismatched"], shuffle=False, collate_fn=data_collator, batch_size=args.bs
            )
    else:
        eval_dataloader = DataLoader(
                tokenized_dataset["validation"], shuffle=False, collate_fn=collate_fn, batch_size=args.bs
            )
    return train_dataloaders, eval_dataloader, sample_nums