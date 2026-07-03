import torch
import numpy as np
import copy
import time
import tracemalloc

def get_proxy_dict(fed_args, global_dict):
    opt_proxy_dict = None
    proxy_dict = None
    if fed_args.alg in ['fedadagrad', 'fedyogi', 'fedadam']:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.ones_like(global_dict[key]) * fed_args.fedopt_tau**2
    elif fed_args.alg == 'fedavgm':
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
    return proxy_dict, opt_proxy_dict

def get_auxiliary_dict(fed_args, global_dict):

    if fed_args.alg in ['scaffold']:
        global_auxiliary = {}               # c in SCAFFOLD
        for key in global_dict.keys():
            global_auxiliary[key] = torch.zeros_like(global_dict[key])
        auxiliary_model_list = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]    # c_i in SCAFFOLD
        auxiliary_delta_dict = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]    # delta c_i in SCAFFOLD

    else:
        global_auxiliary = None
        auxiliary_model_list = [None]*fed_args.num_clients
        auxiliary_delta_dict = [None]*fed_args.num_clients

    return global_auxiliary, auxiliary_model_list, auxiliary_delta_dict


def ours_init(args, init_dict, model):

    param_names = []
    for key in init_dict.keys():
        if 'lora_A' in key:
            param_names.append(key.replace('.lora_A.weight', ''))
        elif 'lora_B' in key:
            param_names.append(key.replace('.lora_B.weight', ''))
    
    start_time = time.time()
    tracemalloc.start()
    
    param_names = set(param_names)
    scale = args.peft_lora_alpha / args.peft_lora_r
    for key in param_names:
        w = model.state_dict()[key+'.weight']
        r = args.peft_lora_r
        V, S, Uh = torch.linalg.svd(w, full_matrices=False)
        Vr = V[:, :r]
        Sr = S[: r]
        Sr /= scale
        Uhr = Uh[: r]

        # Vr = V[:, -r:]
        # Sr = S[-r:]
        # Sr /= scale
        # Uhr = Uh[-r:]

        B2 = torch.diag(torch.sqrt(Sr)) @ Uhr
        B1 = Vr @ torch.diag(torch.sqrt(Sr))

        init_dict[key+'.lora_B.weight'] = B1
        init_dict[key+'.lora_A.weight'] = B2
        
        temp = model.state_dict()[key+'.weight'] - B1@B2*scale
        model.state_dict()[key+'.weight'].data.copy_(temp)
          
    return init_dict, model
    

def statistic_FNorm(global_dict, local_dict_list, client_nums):
    param_names = []
    for key in global_dict.keys():
        if 'lora_A' in key and key.replace('.lora_A.weight', '') not in param_names:
            param_names.append(key.replace('.lora_A.weight', ''))
        elif 'lora_B' in key and key.replace('.lora_B.weight', '') not in param_names:
            param_names.append(key.replace('.lora_B.weight', ''))
    
    #global_fnorm
    global_fnorm = []
    # param_names = set(param_names)
    for name in param_names:
        if 'classifier' in name:
            delt_w = global_dict[name]
        else:
            g_lora_a = global_dict[name+'.lora_A.weight']
            g_lora_b = global_dict[name+'.lora_B.weight']
            delt_w = g_lora_b @ g_lora_a * 2
        temp_norm = np.linalg.norm(delt_w.cpu().numpy())
        global_fnorm.append(temp_norm)
    global_fnorm = np.array(global_fnorm)  
    
    local_norms = {}
    for client in range(client_nums):
        local_norms[client] = [] 

    for name in param_names:
        for client in range(client_nums):
            if 'classifier' in name:
                delt_w = local_dict_list[client][name]
            else:
                g_lora_a = local_dict_list[client][name+'.lora_A.weight']
                g_lora_b = local_dict_list[client][name+'.lora_B.weight']
                delt_w = g_lora_b @ g_lora_a * 2
            temp_norm = np.linalg.norm(delt_w.cpu().numpy())
            local_norms[client].append(temp_norm)

    for key in range(client_nums):
        local_norms[key] = np.array(local_norms[key])
    
    local_norms_final = np.stack([local_norms[key] for key in range(client_nums)], axis=0)
    return global_fnorm, local_norms_final



def statistic_lora_FNorm(global_dict, local_dict_list, client_nums):
    param_names = []
    global_fnorm = []
    for key in global_dict.keys():
        temp_norm = np.linalg.norm(global_dict[key].cpu().numpy())
        global_fnorm.append(temp_norm)
    global_fnorm = np.array(global_fnorm)  
    
    local_norms = {}
    for client in range(client_nums):
        local_norms[client] = [] 

    for key in global_dict.keys():
        for client in range(client_nums):
            temp_norm = np.linalg.norm(local_dict_list[client][key].cpu().numpy())
            local_norms[client].append(temp_norm)

    for key in range(client_nums):
        local_norms[key] = np.array(local_norms[key])
    
    local_norms_final = np.stack([local_norms[key] for key in range(client_nums)], axis=0)
    return global_fnorm, local_norms_final


def statistic_Difference(global_dict, local_dict_list, sample_num_list, clients_nums):
    param_names = []
    sample_this_round = sum(sample_num_list)
    clients_this_round = range(clients_nums)
    for key in global_dict.keys():
        if 'lora_A' in key:
            param_names.append(key.replace('.lora_A.weight', ''))
        elif 'lora_B' in key:
            param_names.append(key.replace('.lora_B.weight', ''))
    
    #global_fnorm
    global_fnorm = []
    param_names = set(param_names)
    for key in param_names:
        if 'classifier' not in key:
            delt_w_g = sum([local_dict_list[client][key+'.lora_B.weight'] @ local_dict_list[client][key+'.lora_A.weight'] * sample_num_list[client] / sample_this_round for client in clients_this_round])
            A_g =  sum([local_dict_list[client][key+'.lora_A.weight'] * sample_num_list[client] / sample_this_round for client in clients_this_round])
            B_g =  sum([local_dict_list[client][key+'.lora_B.weight'] * sample_num_list[client] / sample_this_round for client in clients_this_round])
            BA_g = B_g @ A_g
            temp_norm = np.linalg.norm((delt_w_g-BA_g).cpu().numpy())
            global_fnorm.append(temp_norm)

    return np.mean(np.array(global_fnorm))


def global_aggregate(args, global_dict, local_dict_list, sample_num_list, clients_nums, round_idx=None, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None, init_dict=None):

    sample_num_list = [1 for i in range(clients_nums)]
    sample_this_round = sum(sample_num_list)
    clients_this_round = range(clients_nums)

    if args.alg == ['fedSVD']:
        global_fnorm = []
        param_names = []
        scale = args.peft_lora_alpha / args.peft_lora_r
        for key in global_dict.keys():
            if 'lora_A' in key:
                param_names.append(key.replace('.lora_A.weight', ''))
            elif 'lora_B' in key:
                param_names.append(key.replace('.lora_B.weight', ''))
            elif 'classifier' in key:
                param_names.append(key)
        param_names = set(param_names)

        for key in param_names:
                        
            if 'classifier' in key:
                global_dict[key] = sum([local_dict_list[client][key]  * sample_num_list[client] / sample_this_round])
            else:
                w_g = sum([local_dict_list[client][key+'.lora_B.weight'] @ local_dict_list[client][key+'.lora_A.weight']*scale  * sample_num_list[client] / sample_this_round])
                
                w_g -= init_dict[key+'.lora_B.weight'] @ init_dict[key+'.lora_A.weight'] * scale 

                r = args.peft_lora_r

                V, S, Uh = torch.linalg.svd(w_g, full_matrices=False)
                Vr = V[:, : r]
                Sr = S[: r]
                Sr /= scale
                Uhr = Uh[: r]

                B2 = torch.diag(torch.sqrt(Sr)) @ Uhr
                B1 = Vr @ torch.diag(torch.sqrt(Sr))

                global_dict[key+'.lora_B.weight'] = B1
                global_dict[key+'.lora_A.weight'] = B2

                LR_w = B1 @ B2

                temp_norm = np.linalg.norm((LR_w-w_g).cpu().numpy())
                global_fnorm.append(temp_norm)
        
        lr_dif = np.mean(np.array(global_fnorm))
    
    elif args.alg == 'svd_agg':
        param_names = []
        scale = args.peft_lora_alpha / args.peft_lora_r
        for key in global_dict.keys():
            if 'lora_A' in key:
                param_names.append(key.replace('.lora_A.weight', ''))
            elif 'lora_B' in key:
                param_names.append(key.replace('.lora_B.weight', ''))
            elif 'classifier' in key:
                param_names.append(key)
        param_names = set(param_names)

        for key in param_names:
                
            if 'classifier' in key:
                global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round  for client in clients_this_round])
            else:
                for i, client in enumerate(clients_this_round):
                    w_i = local_dict_list[client][key+'.lora_B.weight'] @ local_dict_list[client][key+'.lora_A.weight']*scale 
                    r = args.peft_lora_r
                    V, S, Uh = torch.linalg.svd(w_i, full_matrices=False)
                    Vr = V[:, : r]
                    Sr = S[: r]
                    Sr /= scale
                    Uhr = Uh[: r]
                    B2 = torch.diag(torch.sqrt(Sr)) @ Uhr
                    B1 = Vr @ torch.diag(torch.sqrt(Sr))
                    if i==0:
                        lora_a = B2 * sample_num_list[client] / sample_this_round 
                        lora_b = B1 * sample_num_list[client] / sample_this_round 
                    else:
                        lora_a += B2 * sample_num_list[client] / sample_this_round 
                        lora_b += B1 * sample_num_list[client] / sample_this_round 
                
                global_dict[key+'.lora_B.weight'] = lora_b
                global_dict[key+'.lora_A.weight'] = lora_a
                lr_dif = 0
    
    elif args.alg in ['adaptive_agg']:
        param_names = []
        scale = args.peft_lora_alpha / args.peft_lora_r
        for key in global_dict.keys():
            if 'lora_A' in key:
                param_names.append(key.replace('.lora_A.weight', ''))
            elif 'lora_B' in key:
                param_names.append(key.replace('.lora_B.weight', ''))
            elif 'classifier' in key:
                param_names.append(key)
        param_names = set(param_names)

        for name in param_names:
            if 'classifier' in name:
                client_weights = []
                for client in range(clients_nums):
                    delt_w = local_dict_list[client][name]
                    temp_norm = np.linalg.norm(delt_w.cpu().numpy())
                    client_weights.append(temp_norm)
                client_weights = np.array(client_weights)
                client_weights /= client_weights.sum()
                global_dict[name] = sum([local_dict_list[client][name] * client_weights[client] for client in clients_this_round])
                #global_dict[name]  = sum([local_dict_list[client][name] * sample_num_list[client] / sample_this_round  for client in clients_this_round])

            else:
                client_weights = []
                for client in range(clients_nums):
                    g_lora_a = local_dict_list[client][name+'.lora_A.weight']
                    g_lora_b = local_dict_list[client][name+'.lora_B.weight']
                    delt_w = g_lora_b @ g_lora_a
                    temp_norm = np.linalg.norm(delt_w.cpu().numpy())
                    client_weights.append(temp_norm)
                client_weights = np.array(client_weights)
                client_weights /= client_weights.sum()
                global_dict[name+'.lora_A.weight'] = sum([local_dict_list[client][name+'.lora_A.weight'] * client_weights[client] for client in clients_this_round])
                global_dict[name+'.lora_B.weight'] = sum([local_dict_list[client][name+'.lora_B.weight'] * client_weights[client] for client in clients_this_round])

            lr_dif = 0
    elif args.alg == 'explore':
        param_names = []
        delta_param_dicts = {}
        scale = args.peft_lora_alpha / args.peft_lora_r
        for key in global_dict.keys():
            if 'lora_A' in key:
                param_names.append(key.replace('.lora_A.weight', ''))
            elif 'lora_B' in key:
                param_names.append(key.replace('.lora_B.weight', ''))
            elif 'classifier' in key:
                param_names.append(key)
        param_names = set(param_names)

        for key in param_names:
            if 'classifier' in key:
                delta_param_dicts[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
            else:
                delta_param_dicts[key]= sum([local_dict_list[client][key+'.lora_B.weight'] @ local_dict_list[client][key+'.lora_A.weight'] * scale * sample_num_list[client] / sample_this_round for client in clients_this_round])
        
        return  delta_param_dicts

    elif args.alg == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
            proxy_dict[key] = args.fedopt_beta1 * proxy_dict[key] + (1 - args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key] = global_dict[key] + proxy_dict[key]
        lr_dif = 0
    
    elif args.alg == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+args.fedopt_tau)
        lr_dif = 0
    elif args.alg == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = args.fedopt_beta1 * proxy_dict[key] + (1 - args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1-args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
            global_dict[key] += args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+args.fedopt_tau)

        lr_dif = 0
    
    elif args.alg == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = args.fedopt_beta1 * proxy_dict[key] + (1 - args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            opt_proxy_dict[key] = args.fedopt_beta2*param + (1-args.fedopt_beta2)*torch.square(proxy_dict[key])
            global_dict[key] += args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+args.fedopt_tau)
        lr_dif = 0
    
    elif args.alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / clients_nums
        lr_dif = 0

    else:   # Normal dataset-size-based aggregation
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        lr_dif = 0

    return global_dict, lr_dif
