from torch.optim import AdamW, SGD
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import copy
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import torch

class FedAvgTrainer:
    def __init__(self, model, lr_head, lr_lora, train_loader, weight_decay, rounds, device, local_update_step):
        
        self.model = model

        for name, param in self.model.named_parameters():
            if 'lora_A.default.weight' in name:
                param.requires_grad = False

        head_param = list(map(id, self.model.classifier.parameters()))

        others_param = filter(lambda p: id(p) not in head_param, self.model.parameters()) 

        self.train_loader = train_loader
        self.device = device

        # self.optimizer = SGD([
        #     {"params": self.model.classifier.parameters(), "lr": lr_head},
        #     {"params": others_param, "lr": lr_lora}
        # ],  momentum=0.9, weight_decay=weight_decay)

       

        self.optimizer = AdamW([
            {"params": self.model.classifier.parameters(), "lr": lr_head},
            {"params": others_param, "lr": lr_lora}
        ], weight_decay=weight_decay)

        # self.lr_scheduler = get_linear_schedule_with_warmup(
        #     optimizer=self.optimizer,
        #     num_warmup_steps=0.06 * (len(self.train_loader) * rounds),
        #     num_training_steps=(len(self.train_loader) * rounds))

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0.03 * (len(self.train_loader) * rounds),
            num_training_steps=(len(self.train_loader) * rounds))

        self.local_update_step = local_update_step

        
    
    def train(self):
        self.model.train()
        loss_v = 0
        for i in tqdm(range(self.local_update_step)):
            batch = next(iter(self.train_loader))
            batch.to(self.device)
            outputs = self.model(**batch)
            self.optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            loss_v += loss.detach().item()

        return loss_v / len(self.train_loader)

    
    def get_model_parameters(self):
        return copy.deepcopy(get_peft_model_state_dict(self.model))

    def set_model_parameters(self, global_dict):
        set_peft_model_state_dict(self.model, global_dict)


class FullTrainer:
    def __init__(self, model, lr_head, lr_lora, train_loader, weight_decay, rounds, device, local_update_step):
        
        self.model = model

        head_param = list(map(id, self.model.classifier.parameters()))

        others_param = filter(lambda p: id(p) not in head_param, self.model.parameters()) 

        self.train_loader = train_loader
        self.device = device

        self.optimizer = AdamW([
            {"params": self.model.classifier.parameters(), "lr": lr_head},
            {"params": others_param, "lr": lr_lora}
        ], weight_decay=weight_decay)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0.03 * (len(self.train_loader) * rounds),
            num_training_steps=(len(self.train_loader) * rounds))

        self.local_update_step = local_update_step

        
    def train(self):
        self.model.train()
        loss_v = 0
        for i in tqdm(range(self.local_update_step)):
            batch = next(iter(self.train_loader))
            batch.to(self.device)
            outputs = self.model(**batch)
            self.optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            loss_v += loss.detach().item()
        return loss_v / len(self.train_loader)

    
    def get_model_parameters(self):
        return copy.deepcopy(self.model.state_dict())

    def set_model_parameters(self, global_dict):
        self.model.load_state_dict(global_dict)



class ScaffoldTrainer:
    def __init__(self, model, lr_head, lr_lora, train_loader, weight_decay, rounds, device, global_state, local_auxiliary, global_auxiliary, local_update_step, max_steps=10):
        
        self.model = model

        head_param = list(map(id, self.model.classifier.parameters()))

        others_param = filter(lambda p: id(p) not in head_param, self.model.parameters()) 

        self.train_loader = train_loader
        self.device = device

        self.optimizer = AdamW([
            {"params": self.model.classifier.parameters(), "lr": lr_head},
            {"params": others_param, "lr": lr_lora}
        ], weight_decay=weight_decay)

        self.lr = lr_lora

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0.03 * (len(self.train_loader) * rounds),
            num_training_steps=(len(self.train_loader) * rounds))

        self.local_update_step = local_update_step

        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)
        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
        
        self.max_steps = max_steps
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in copy.deepcopy(get_peft_model_state_dict(self.model)).items():
                auxiliary_new_para[name] = (self.global_state[name] - param) / (self.max_steps * self.lr) - self.correction[name]
                auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para

    def train(self):
        self.model.train()
        loss_v = 0
        for i in tqdm(range(self.local_update_step)):
            batch = next(iter(self.train_loader))
            batch.to(self.device)
            outputs = self.model(**batch)
            self.optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            loss_v += loss.detach().item()

            model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
            for name in model_para.keys():
                model_para[name] -= self.lr * self.correction[name]
            set_peft_model_state_dict(self.model, model_para)

        return loss_v / len(self.train_loader)

    
    def get_model_parameters(self):
        return copy.deepcopy(get_peft_model_state_dict(self.model))

    def set_model_parameters(self, global_dict):
        set_peft_model_state_dict(self.model, global_dict)



class FedProxTrainer:
    def __init__(self, model, lr_head, lr_lora, train_loader, weight_decay, rounds, device, mu, local_update_step):
        
        self.model = model

        head_param = list(map(id, self.model.classifier.parameters()))

        others_param = filter(lambda p: id(p) not in head_param, self.model.parameters()) 

        self.train_loader = train_loader
        self.device = device

        self.optimizer = AdamW([
            {"params": self.model.classifier.parameters(), "lr": lr_head},
            {"params": others_param, "lr": lr_lora}
        ], weight_decay=weight_decay)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0.03 * (len(self.train_loader) * rounds),
            num_training_steps=(len(self.train_loader) * rounds))

        self.mu = mu

        self.local_update_step = local_update_step

    def train(self):
        
        global_state = copy.deepcopy(get_peft_model_state_dict(self.model))

        self.model.train()
        loss_v = 0
        reg_loss_v = 0

        for i in tqdm(range(self.local_update_step)):
            batch = next(iter(self.train_loader))
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            
            reg_loss = 0
            for name, param in get_peft_model_state_dict(self.model).items():
                reg_loss  += self.mu / 2 * torch.norm(param - global_state[name]) ** 2

            loss_v += loss.detach().item()
            reg_loss_v += reg_loss.detach().item()

            loss += reg_loss
            
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            loss_v += loss.detach().item()

        return loss_v / len(self.train_loader), reg_loss_v/len(self.train_loader), (loss_v + reg_loss_v)/len(self.train_loader)

    
    def get_model_parameters(self):
        return copy.deepcopy(get_peft_model_state_dict(self.model))

    def set_model_parameters(self, global_dict):
        set_peft_model_state_dict(self.model, global_dict)


class RegulationTrainer:
    def __init__(self, model, lr_head, lr_lora, train_loader, weight_decay, rounds, device, mu, scale, r):
        
        self.model = model

        head_param = list(map(id, self.model.classifier.parameters()))

        others_param = filter(lambda p: id(p) not in head_param, self.model.parameters()) 

        # self.optimizer = SGD([
        #     {"params": self.model.classifier.parameters(), "lr": lr_head},
        #     {"params": others_param, "lr": lr_lora}
        # ],  momentum=0.9, weight_decay=weight_decay)

        # self.optimizer = optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr_lora, momentum=0.9,
        #                       weight_decay=weight_decay)

        self.train_loader = train_loader
        self.device = device

        # self.lr_scheduler = get_linear_schedule_with_warmup(
        #     optimizer=self.optimizer,
        #     num_warmup_steps=0.06 * (len(self.train_loader) * rounds),
        #     num_training_steps=(len(self.train_loader) * rounds))

        self.optimizer = AdamW([
            {"params": self.model.classifier.parameters(), "lr": lr_head},
            {"params": others_param, "lr": lr_lora}
        ], weight_decay=weight_decay)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0.03 * (len(self.train_loader) * rounds),
            num_training_steps=(len(self.train_loader) * rounds))


        self.mu = mu
        self.scale = scale
        self.r = r 
        self.reg_base = self.set_reg_base()

        param_names = []
        init_dict = copy.deepcopy(get_peft_model_state_dict(self.model))
        for key in init_dict.keys():
            if 'lora_A' in key:
                param_names.append(key.replace('.lora_A.weight', ''))
            elif 'lora_B' in key:
                param_names.append(key.replace('.lora_B.weight', ''))
        self.param_names = param_names
    
    def set_reg_base(self):
        param_names = []
        init_dict = copy.deepcopy(get_peft_model_state_dict(self.model))
        for key in init_dict.keys():
            if 'lora_A' in key:
                param_names.append(key.replace('.lora_A.weight', ''))
            elif 'lora_B' in key:
                param_names.append(key.replace('.lora_B.weight', ''))
    
        param_names = set(param_names)
        for key in param_names:
            w = self.model.state_dict()[key+'.weight']
            r = self.r
            V, S, Uh = torch.linalg.svd(w, full_matrices=False)
            Vr = V[:, : r]
            Sr = S[: r]
            Sr /= self.scale
            Uhr = Uh[: r]

            B2 = torch.diag(torch.sqrt(Sr)) @ Uhr
            B1 = Vr @ torch.diag(torch.sqrt(Sr))

            init_dict[key+'.lora_B.weight'] = B1
            init_dict[key+'.lora_A.weight'] = B2
        return init_dict
    
    def train(self):

        self.model.train()
        loss_v = 0
        reg_loss_v = 0
        for step, batch in enumerate(tqdm(self.train_loader)):
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss

            reg_loss = 0

            # for name, param in self.model.named_parameters():
            #     name = name.replace(".default", "") 
            #     if not param.requires_grad or name not in self.reg_base.keys():
            #         continue
            #     else:
            #         reg_loss += self.mu /2 * torch.norm(param) **2


            t_dict = copy.deepcopy(get_peft_model_state_dict(self.model))
            for name in self.param_names:
                delt_w = t_dict[name+'.lora_B.weight'] @ t_dict[name+'.lora_A.weight'] * self.scale
                reg_loss += self.mu /2 * torch.norm(delt_w) **2
            
            loss_v += loss.detach().item()
            reg_loss_v += reg_loss.detach().item()

            loss += reg_loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return loss_v / len(self.train_loader), reg_loss_v/len(self.train_loader), (loss_v + reg_loss_v)/len(self.train_loader)

    
    def get_model_parameters(self):
        return copy.deepcopy(get_peft_model_state_dict(self.model))

    def set_model_parameters(self, global_dict):
        set_peft_model_state_dict(self.model, global_dict)


class FFALORATrainer:
    def __init__(self, model, lr_head, lr_lora, train_loader, weight_decay, rounds, device, local_update_step):
        
        self.model = model

        for name, param in self.model.named_parameters():
            if 'lora_A.default.weight' in name:
                param.requires_grad = False

        head_param = list(map(id, self.model.classifier.parameters()))

        others_param = filter(lambda p: id(p) not in head_param, self.model.parameters()) 


        self.train_loader = train_loader
        self.device = device

        self.local_update_step = local_update_step


        self.optimizer = AdamW([
            {"params": self.model.classifier.parameters(), "lr": lr_head},
            {"params": others_param, "lr": lr_lora}
        ], weight_decay=weight_decay)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0.03 * (len(self.train_loader) * rounds),
            num_training_steps=(len(self.train_loader) * rounds))
    
    def train(self):
        self.model.train()
        loss_v = 0

        for i in tqdm(range(self.local_update_step)):
            batch = next(iter(self.train_loader))
            batch.to(self.device)
            outputs = self.model(**batch)
            self.optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            loss_v += loss.detach().item()
        return loss_v / len(self.train_loader)

    
    def get_model_parameters(self):
        return copy.deepcopy(get_peft_model_state_dict(self.model))

    def set_model_parameters(self, global_dict):
        set_peft_model_state_dict(self.model, global_dict)


def build_local_trainers(alg, model, lr_head, lr_lora, train_loaders, weight_decay, rounds, device, mu, scale, r, local_update_step, global_dict=None, local_auxiliary=None, global_auxiliary=None):
    trainers = []
    i=0
    for loader in train_loaders:
        if alg in ['fedavg', 'fedSVD', 'explore', 'adaptive_agg', 'svd_agg', 'fedavgm', 'fedyogi', 'fedadagrad', 'fedadam']:
            trainer = FedAvgTrainer(copy.deepcopy(model), lr_head, lr_lora, loader, weight_decay, rounds, device, local_update_step)
            trainers.append(trainer)
        elif alg == 'fedreg':
            trainer = RegulationTrainer(copy.deepcopy(model), lr_head, lr_lora, loader, weight_decay, rounds, device, mu, scale, r)
            trainers.append(trainer)
        elif alg == 'ffalora':
            trainer = FFALORATrainer(copy.deepcopy(model), lr_head, lr_lora, loader, weight_decay, rounds, device, local_update_step)
            trainers.append(trainer)
        elif alg == 'fedprox':
            trainer = FedProxTrainer(copy.deepcopy(model), lr_head, lr_lora, loader, weight_decay, rounds, device, mu, local_update_step)
            trainers.append(trainer)
        elif alg == 'scaffold':
            trainer = ScaffoldTrainer(copy.deepcopy(model), lr_head, lr_lora, loader, weight_decay, rounds, device,  global_dict, local_auxiliary[i], global_auxiliary, local_update_step)
            trainers.append(trainer)
        elif alg == 'ft':
            trainer = FullTrainer(copy.deepcopy(model), lr_head, lr_lora, loader, weight_decay, rounds, device, local_update_step)
            trainers.append(trainer)
        else:
            print('error alg')
        i +=1
    
    return trainers
        
