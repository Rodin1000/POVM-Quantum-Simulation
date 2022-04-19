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
    parser.add_argument("--mini_batch_size", type=int, default=int(100))
    parser.add_argument("--accumulation_step", type=int, default=int(1))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--eval_nb_samples", type=int, default=int(1e4))
    parser.add_argument("--evaluate", type=int, default=True)
    parser.add_argument("--max_step", type=int, default=120)
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


def load_model(model, parallel=True, path=None):
    if path == None:
        return model
    previous_model_path = path
    best_dict = torch.load(previous_model_path)

    if parallel == True:
        new_state_dict = OrderedDict()
        for k, v in best_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(best_dict)
    return model

def train_model(model, model_copy, circuit, povm, train_config, device, B_factor, Jx_factor, Jy_factor, Jz_factor, \
                gamma = 1, label = 1, total_iteration = 100, dt = 0.001, start_from = 0):
    g_factor = Jy_factor # delete it later####
    operator_2qbit = Jx_factor * povm.XX_com + Jy_factor * povm.YY_com + Jz_factor * povm.ZZ_com
    operator_1qbit = B_factor * povm.ZI_com + gamma * povm.Minus_gate - gamma/2. * povm.Plus_minus_anti
    operator = operator_2qbit+operator_1qbit/2
    operator = torch.tensor(np.real(operator)).float().to(device)
    train_model.operat = operator
    num_rows = int(np.sqrt(circuit_config.nb_qbits))
    num_columns = int(np.sqrt(circuit_config.nb_qbits))
   
    if torch.cuda.device_count() >= 1 and train_config.device == "cuda":
       logging.info("Let's use number of {} GPUs".format(torch.cuda.device_count()))
       model = torch.nn.DataParallel(model)
       model_copy = torch.nn.DataParallel(model_copy)

    writer = SummaryWriter(comment="_" + train_config.save_dir)
    fs = open(train_config.save_dir+"/sigmas.txt", "a+")
    fs.flush()
    accumulation_step = train_config.accumulation_step
    lr = train_config.lr
    for i in range(start_from, total_iteration+start_from):
        # Optimizer and dataloader
        model_copy.load_state_dict(model.state_dict()) # copy state
        optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=np.array([80]), gamma=0.5)
    
        # Variable initializations
        start_time = time()
        
        global_step = 0
        f2 = open(train_config.save_dir+"/record_{}.txt".format(i), 'a+')
        
        while global_step < train_config.max_step:
    
            global_step += 1
            
            # batch from p_exact
            # flip coef from flip2(batch operator)
            # combine coef according to flip
            
            ### this is for forward loss
            batch, logP_samples = model(forward_type="sample")
            batch = batch.to(device)
            logP_samples = logP_samples.to(device)
            
            p_new1 = utils.reverse_samples_lindblad_2d_pbc(batch, operator, 4, model, int(np.sqrt(circuit_config.nb_qbits)), int(np.sqrt(circuit_config.nb_qbits)), True)
            p_new1 = torch.exp(utils.sum_log_p(batch, model)) - dt * p_new1
            p_new2 = utils.reverse_samples_lindblad_2d_pbc(batch, operator, 4, model_copy, int(np.sqrt(circuit_config.nb_qbits)), int(np.sqrt(circuit_config.nb_qbits)), False)
            p_new2 =  torch.exp(model_copy('logP', batch)) + dt * p_new2
            loss = utils.forward_backward_trapozoid_L1(batch, logP_samples, p_new1, p_new2)
            loss = loss / accumulation_step
            loss.backward()
    
            ### accumulation step
            if global_step % accumulation_step == 0:
                optim.step()
                optim.zero_grad()
                scheduler.step()
            if global_step % 1 == 0:
                for item in [global_step, loss.item()]:
                    f2.write("%s " % item)
                f2.write('\n')
                logging.info("step {}".format(global_step))
                logging.info("stochastic loss {}".format(loss.item()))
                logging.info("")
    
            writer.add_scalar("{}_learning_rate".format(i), scheduler.get_lr(), global_step=global_step)
            writer.add_scalar("{}_loss".format(i), loss, global_step=global_step)

        ZI = torch.from_numpy(ncon((povm.ZI.reshape(2,2,2,2), povm.Nt, povm.Nt), ([1,2,3,4],[-1,3,1],[-2,4,2])).real.astype(np.float32)).to(device)
        z = utils.compute_energy_gpu_pbc(ZI, ZI, circuit_config.nb_qbits, train_config.accumulation_step*500, train_config.mini_batch_size, model)
        for item in [i, z[0].cpu().item()/circuit_config.nb_qbits, z[1].cpu().item()/circuit_config.nb_qbits]:
            fs.write("%s " % item)
        fs.write('\n')
        fs.flush()
        if train_config.save_dir:
            logging.info("Saving model in {}".format(train_config.save_dir))
            model = model.cpu()
            torch.save(model.state_dict(),
                       os.path.join("", train_config.save_dir, "{}_model.pt".format(i)))
            model.to(device)
    fs.close()

    end_time = time()
    logging.info("\nTraining took {:.3f} seconds".format(end_time - start_time))





### training
    
# Cuda semantics
model_config, train_config, circuit_config = get_config()
if train_config.device == "cuda" and torch.cuda.is_available():
    device = torch.device('cuda')
    logging.info("cuda available")
else:
    device = torch.device('cpu')
start_time = time()
start_from = 0
model_path = None
model = 'QCModel'
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



if model_path:
    model = load_model(model, True, model_path)
    model_copy = load_model(model_copy, True, model_path)
print(model.count_params())
circuit = get_circuit(circuit_config)
dt = 0.008
train_model(model, model_copy, circuit, povm, train_config, device, B_factor = 0, Jx_factor = 0.9, Jy_factor = 1.6, Jz_factor = 1.0, total_iteration = 10, dt = dt, start_from = start_from)
sigmas = np.loadtxt(train_config.save_dir+"/sigmas.txt")

plt.figure()
plt.errorbar(sigmas[:, 0] * dt * 2, sigmas[:, 1], sigmas[:, 2])
plt.ylabel(r'$\langle\sigma_z\rangle$')
plt.xlabel('t')
plt.savefig(os.path.join(train_config.save_dir, 'sigmaz.png'), dpi=200)

end_time = time()
logging.info("\nTotal training took {:.3f} seconds".format(end_time - start_time))