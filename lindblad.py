from argparse import ArgumentParser
from utils import POVM, QCModelConfig, TrainConfig, CircuitConfig, get_circuit, ncon
import torch
import sys
from torch.optim import Adam
from model import utils
from model import QCModel, flip2
from torch.utils.data import DataLoader
from time import time
import os
from tensorboardX import SummaryWriter
from itertools import product
import logging
import copy
import numpy as np
import scipy as sp
import tensorly as tl
from collections import OrderedDict
from tensorly.decomposition import matrix_product_state
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import pyplot as plt
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_config(nb_qbits=0, save_dir=""):
    parser = ArgumentParser()

    # ============================== MODEL OPTIONS ====================================
    parser.add_argument("--nb_measurements", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--nb_encoder_layers", type=int, default=1)
    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--intermediate_dim", type=int, default=16)
    parser.add_argument("--hidden_act", type=str, default="gelu")
    parser.add_argument("--positional_encoding_dropout", type=float, default=0)
    parser.add_argument("--max_position_embeddings", type=int, default=100)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0)
    parser.add_argument("--attention_dropout_prob", type=float, default=0)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--layer_norm_type", type=str, default="annotated")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)

    # ========================== TRAINING & EVALUATION OPTIONS ==================================

    parser.add_argument("--exponent_index", type=int, default=0)
    parser.add_argument("--nb_samples", type=int, default=int(1e5))
    parser.add_argument("--sampling_batch_size", type=int, default=int(1e3))
    parser.add_argument("--mini_batch_size", type=int, default=int(10000))
    parser.add_argument("--accumulation_step", type=int, default=int(1))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--eval_nb_samples", type=int, default=int(1e4))
    parser.add_argument("--evaluate", type=int, default=True)
    parser.add_argument("--max_step", type=int, default=500)
    parser.add_argument("--lr", type=float, default="1e-2")
    parser.add_argument("--beta", type=float, default="10")
    parser.add_argument("--tau", type=float, default="0.1")
    parser.add_argument("--final_state", type=str, default="Graph")
    parser.add_argument("--writer", type=str, default="tensorboardX")
    parser.add_argument("--data_random_seed", type=int, default=13)
    parser.add_argument("--model_random_seed", type=int, default=-1)

    # ================================= CIRCUIT OPTIONS ====================================

    parser.add_argument("--nb_qbits", type=int, default=4)
    parser.add_argument("--circuit_depth", type=int, default=1)
    parser.add_argument("--povm", type=str, default="4Pauli")
    parser.add_argument("--circuit_file", type=str, default="")
    parser.add_argument("--circuit_type", type=str, default="basic")
    parser.add_argument("--initial_product_state", type=str, default="0", help="The options are 0, 1, +, -, r, l")

    args = parser.parse_args()

    if nb_qbits:
        args.nb_qbits = nb_qbits

    args.save_dir = "./results/nb_qbits_{}_".format(args.nb_qbits) + args.save_dir

    qc_model_config = QCModelConfig.from_parsed_args(args)
    train_config = TrainConfig.from_parsed_args(args)
    circuit_config = CircuitConfig.from_parsed_args(args)

    if train_config.save_dir:
        parent_dir = ""
        os.makedirs(os.path.join(parent_dir, args.save_dir), exist_ok=True)
        qc_model_config.to_json_file(os.path.join(parent_dir, args.save_dir, "qc_model_config.json"))
        train_config.to_json_file(os.path.join(parent_dir, args.save_dir, "train_config.json"))
        circuit_config.to_json_file(os.path.join(parent_dir, args.save_dir, "circuit_config.json"))
        file_handler = logging.FileHandler(os.path.join(parent_dir, args.save_dir, "log.txt"), mode="w")
        os.system("cp ./*.sh " + os.path.join(parent_dir, args.save_dir))
        os.system("cp ./*.py " + os.path.join(parent_dir, args.save_dir))
        os.system("cp -r ./model " + os.path.join(parent_dir, args.save_dir))
        os.system("cp -r ./utils " + os.path.join(parent_dir, args.save_dir))
        logger.addHandler(file_handler)

    return qc_model_config, train_config, circuit_config


def train_model(model, model_copy, circuit, povm, train_config, device, V_factor = 4, g_factor = 2, gamma = 1, label = 1):
   
    if torch.cuda.device_count() >= 1 and train_config.device == "cuda":
       logging.info("Let's use number of {} GPUs".format(torch.cuda.device_count()))
       model = torch.nn.DataParallel(model)
       model_copy = torch.nn.DataParallel(model_copy)

    writer = SummaryWriter(comment="_" + train_config.save_dir)
    
    accumulation_step = train_config.accumulation_step
    
    # Optimizer and dataloader
    model_copy.load_state_dict(model.state_dict()) # copy state
    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=train_config.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=np.array([600])*accumulation_step, gamma=0.5)

    # Set up two-site operator
    operator = V_factor/4. * povm.ZZ_com + g_factor/2. * povm.XI_com + gamma * povm.Minus_gate - gamma/2. * povm.Plus_minus_anti
    operator = torch.tensor(np.real(operator)).float().to(device)
    start_time = time()
    global_step = 0
    f2 = open(train_config.save_dir+"/record_{:.4f}_{}.txt".format(g_factor, label), 'a+')
    
    fbest = open(train_config.save_dir+"/accurate_record_{:.4f}_{}.txt".format(g_factor, label), "a+")
    accurate_loss_old = np.inf
    while global_step < train_config.max_step:

        global_step += 1
        
        
        # batch from p_exact
        # flip coef from flip2(batch operator)
        # combine coef according to flip
        
        ### this is for forward loss
        batch, logP_samples = model(forward_type="sample")
        batch = batch.to(device)
        logP_samples = logP_samples.to(device)
        p_new =  utils.reverse_samples_lindblad_pbc(batch, operator, 4, model)

        p_samples = torch.exp(logP_samples)
        loss = (torch.abs(p_new) * torch.pow(p_samples, -1)).mean()
        loss = loss / accumulation_step
        loss.backward()

        ### accumulation step
        if global_step % accumulation_step == 0:
            optim.step()
            optim.zero_grad()
            scheduler.step()
        if global_step // accumulation_step >= 400 and global_step % (accumulation_step * 5) == 0:
            logging.info("Computing accurate loss")
            Loss_mean = 0.0
            Loss2_mean = 0.0
            num_batch = accumulation_step * 4
            Ns = num_batch * train_config.mini_batch_size
            
            # looks like there is some memory problem of doing this inside a function
            for n in range(num_batch):
                print(n)
                batch, logP_samples = model(forward_type="sample")
                batch = batch.to(device)
                logP_samples = logP_samples.to(device)
                p_new = utils.reverse_samples_lindblad_pbc(batch, operator, 4, model).detach()
                p_samples = torch.exp(logP_samples).detach()
                Loss = (torch.abs(p_new) * torch.pow(p_samples, -1))
                Loss2 = Loss * Loss
                Loss_mean += torch.mean(Loss).item()
                Loss2_mean += torch.mean(Loss2).item()
        
            Loss_mean = Loss_mean / num_batch
            Loss2_mean = Loss2_mean / num_batch
            Err = np.sqrt( (Loss2_mean- Loss_mean**2)/Ns)
            
            accurate_loss = Loss_mean
            accurate_loss_err = Err
            for item in [global_step, accurate_loss, accurate_loss_err]:
                fbest.write("%s " % item)
            fbest.write('\n')
            if accurate_loss < accurate_loss_old:
                accurate_loss_old = accurate_loss
                if os.path.exists(os.path.join("", train_config.save_dir, "best_model_{:.4f}_{}.pt".format(g_factor, label))):
                    os.remove(os.path.join("", train_config.save_dir, "best_model_{:.4f}_{}.pt".format(g_factor, label)))
                torch.save(model.state_dict(), os.path.join("", train_config.save_dir, "best_model_{:.4f}_{}.pt".format(g_factor, label)))
        if global_step % 1 == 0:
            for item in [global_step, loss.item()]:
                f2.write("%s " % item)
            f2.write('\n')
            logging.info("step {}".format(global_step))
            logging.info("stochastic loss {}".format(loss.item()))
            logging.info("")

        writer.add_scalar("learning_rate", scheduler.get_lr(), global_step=global_step)
        writer.add_scalar("loss", loss, global_step=global_step)


    end_time = time()
    logging.info("\nTraining took {:.3f} seconds".format(end_time - start_time))

    if train_config.save_dir:
        logging.info("Saving model in {}".format(train_config.save_dir))
        model = model.cpu()
        torch.save(model.state_dict(),
                   os.path.join("", train_config.save_dir, "model_{:.4f}_{}.pt".format(g_factor, label)))
        model.to(device)



def load_best_model(g, label, path, model_config):
    previous_model_path = path

    previous_model_config = model_config
    previous_model = QCModel(previous_model_config)
    best_dict = torch.load(os.path.join(previous_model_path,"model_{:.4f}_{}.pt".format(g, label)))
    new_state_dict = OrderedDict()
    for k, v in best_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    previous_model.load_state_dict(new_state_dict)
    return previous_model

    
start_time = time()
X_list = []
X_err = []
Y_list = []
Y_err = []
Z_list = []
Z_err = []
g_list = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                   1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4 ,
                   2.6, 2.8, 3.1, 3.4, 3.7, 4.0, 4.8    ],
                    dtype=np.float32)


# Cuda semantics
model_config, train_config, circuit_config = get_config()
if train_config.device == "cuda" and torch.cuda.is_available():
    device = torch.device('cuda')
    logging.info("cuda available")
else:
    device = torch.device('cpu')


for g in g_list:
    label = 1
    V = 2
    povm = POVM(POVM=circuit_config.povm, Number_qubits=circuit_config.nb_qbits, initial_state=train_config.initial_product_state)
    bias = povm.get_initial_bias(circuit_config.initial_product_state, as_torch_tensor=True)
    
    model = QCModel(model_config, bias,
                         sampling_batch_size=train_config.mini_batch_size,
                         sample_batch=train_config.mini_batch_size,
                         nb_qbits=circuit_config.nb_qbits,
                         eval_nb_samples=train_config.eval_nb_samples).to(device)
    model_copy = QCModel(model_config, bias,
                         sampling_batch_size=train_config.mini_batch_size,
                         sample_batch=train_config.mini_batch_size,
                         nb_qbits=circuit_config.nb_qbits,
                         eval_nb_samples=train_config.eval_nb_samples).to(device)
    print(model.count_params())
    print(model.state_dict())
    circuit = get_circuit(circuit_config)
    
    ### training
    train_model(model, model_copy, circuit, povm, train_config, device, V_factor = V, g_factor = g, label = label)
    
    best_model = load_best_model(g, label, train_config.save_dir, model_config)
    best_model.nb_qbits = circuit.nb_qbits
    best_model.eval_nb_samples = train_config.eval_nb_samples
    best_model.sampling_batch_size = train_config.mini_batch_size
    best_model.sample_batch_total = train_config.mini_batch_size
    best_model.to(device)
    
    XI = torch.from_numpy(ncon((povm.XI.reshape(2,2,2,2), povm.Nt, povm.Nt), ([1,2,3,4],[-1,3,1],[-2,4,2])).real.astype(np.float32)).to(device)
    YI = torch.from_numpy(ncon((povm.YI.reshape(2,2,2,2), povm.Nt, povm.Nt), ([1,2,3,4],[-1,3,1],[-2,4,2])).real.astype(np.float32)).to(device)
    ZI = torch.from_numpy(ncon((povm.ZI.reshape(2,2,2,2), povm.Nt, povm.Nt), ([1,2,3,4],[-1,3,1],[-2,4,2])).real.astype(np.float32)).to(device)
    x = utils.compute_energy_gpu_pbc(XI, XI, circuit_config.nb_qbits, train_config.accumulation_step*100, train_config.mini_batch_size, best_model)
    y = utils.compute_energy_gpu_pbc(YI, YI, circuit_config.nb_qbits, train_config.accumulation_step*100, train_config.mini_batch_size, best_model)
    z = utils.compute_energy_gpu_pbc(ZI, ZI, circuit_config.nb_qbits, train_config.accumulation_step*100, train_config.mini_batch_size, best_model)
    print(x, y, z)
    X_list.append(x[0].cpu().item()/circuit_config.nb_qbits)
    Y_list.append(y[0].cpu().item()/circuit_config.nb_qbits)
    Z_list.append(z[0].cpu().item()/circuit_config.nb_qbits)
    X_err.append(x[1].cpu().item()/circuit_config.nb_qbits)
    Y_err.append(y[1].cpu().item()/circuit_config.nb_qbits)
    Z_err.append(z[1].cpu().item()/circuit_config.nb_qbits)
    
    np.savetxt(os.path.join(train_config.save_dir, 'sigmas.txt'), np.stack((g_list[:len(X_list)], X_list, Y_list, Z_list, X_err, Y_err, Z_err)))

end_time = time()
logging.info("\nTotal training took {:.3f} seconds".format(end_time - start_time))






