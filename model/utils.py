import torch
import itertools as it
from itertools import product
from utils import ncon
import numpy as np 
import scipy as sp
import tensorly as tl
from tensorly.decomposition import matrix_product_state



def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.relu, "swish": swish}


def vectorize(num_sites, K, model, device=torch.device("cpu")):
    """
    :return: a vector of probability distribution of a model over all the basis elements
    """
    l_basis = np.zeros((K**num_sites, num_sites), dtype=np.int32)
    for i in range(K**num_sites):
      basis_str = np.base_repr(i, base=K, padding=num_sites)[-num_sites:]
      l_basis[i, :] = np.array(list(basis_str), dtype=int)
    l_basis = torch.tensor(l_basis, dtype=torch.long, device=device)
    lP = model(forward_type='logP', seq=l_basis)
    return lP.cpu().detach().numpy()



def compute_observables(obs, site, samples):
    """
    compute observables of an operator obs on site from given samp
    <O> = sum Pi obs_i ~ obs_i for samp
    :return: observable coefficient with respect to samples
    """
    ndim = obs.dim()
    if ndim == 1:
      Coef = obs[samples[:, site[0]]]
    elif ndim == 2:
      Coef = obs[samples[:, site[0]], samples[:, site[1]]]
    else:
      raise NameError("dimension not correct")

    return Coef.squeeze()


def compute_observables_correlation(obs, nb_qbits, num_batch, mini_batch_size, model):
    """
    compute observables of an operator obs on site from given samp
    <O> = sum Pi obs_i ~ obs_i for samp
    :return: observable coefficient with respect to samples
    """
    ndim = obs.dim()
    Ns = num_batch * mini_batch_size
    samp, _ = model(forward_type="sample")
    ob_matrix = torch.zeros(nb_qbits, nb_qbits, device=samp.device)
    ob2_matrix = torch.zeros(nb_qbits, nb_qbits, device=samp.device)
    for _ in range(num_batch):
      samp, _ = model(forward_type="sample")
      for i in range(nb_qbits):
          for j in range(i+1, nb_qbits):
              site = [i, j]
              Coef = compute_observables(obs, site, samp)
              Coef2 = Coef * Coef
              ob_matrix[i, j] += torch.mean(Coef)
              ob2_matrix[i, j] += torch.mean(Coef2)

    ob_matrix = ob_matrix / num_batch
    ob2_matrix = ob2_matrix / num_batch
    err_matrix = torch.sqrt((ob2_matrix - ob_matrix * ob_matrix) / Ns)
    ob_matrix = ob_matrix + torch.t(ob_matrix) + torch.eye(nb_qbits, device=samp.device)
    err_matrix = err_matrix + torch.t(err_matrix)

    return ob_matrix, err_matrix


def compute_energy(hl_ob, hlx_ob, Nqubit, samp):
    """
    compute expectation value of Hamiltonian, where Hamiltonian is TFIM formed from hlx_ob and hl_ob
    :return: mean of energy, standard deviation
    """
    Ns = samp.shape[0]
    Coef = compute_observables(hlx_ob, [Nqubit-2,Nqubit-1],samp)
    for i in range(Nqubit-2):
        Coef += compute_observables(hl_ob, [i,i+1], samp)
    Coef2 = Coef * Coef
    Coef_mean = torch.mean(Coef)
    Coef2_mean = torch.mean(Coef2)
    Err = torch.sqrt( (Coef2_mean- Coef_mean**2)/Ns)
    return Coef_mean, Err


def compute_energy_gpu(hl_ob, hlx_ob, Nqubit, num_batch, mini_batch_size, model):
    """
    compute expectation value of Hamiltonian on gpu, where Hamiltonian is TFIM formed from hlx_ob and hl_ob
    :return: mean of energy, standard deviation
    """
    Coef_mean = 0.0
    Coef2_mean = 0.0
    Ns = num_batch * mini_batch_size
    for _ in range(num_batch):
      samp, _ = model(forward_type="sample")
      Coef = compute_observables(hlx_ob, [Nqubit-2,Nqubit-1],samp)
      for i in range(Nqubit-2):
          Coef += compute_observables(hl_ob, [i,i+1], samp)
      Coef2 = Coef * Coef
      Coef_mean += torch.mean(Coef)
      Coef2_mean += torch.mean(Coef2)

    Coef_mean = Coef_mean / num_batch
    Coef2_mean = Coef2_mean / num_batch
    Err = torch.sqrt( (Coef2_mean- Coef_mean**2)/Ns)
    return Coef_mean, Err


def compute_energy_gpu_pbc(hl_ob, hlx_ob, Nqubit, num_batch, mini_batch_size, model):
    """
    compute expectation value of Hamiltonian on gpu, where Hamiltonian is TFIM formed from hlx_ob and hl_ob
    :return: mean of energy, standard deviation
    """
    Coef_mean = 0.0
    Coef2_mean = 0.0
    Ns = num_batch * mini_batch_size
    for _ in range(num_batch):
      samp, _ = model(forward_type="sample")
      Coef = compute_observables(hl_ob, [0,1], samp)
      for i in range(1,Nqubit):
          Coef += compute_observables(hl_ob, [i%Nqubit,(i+1)%Nqubit], samp)
      Coef2 = Coef * Coef
      Coef_mean += torch.mean(Coef)
      Coef2_mean += torch.mean(Coef2)

    Coef_mean = Coef_mean / num_batch
    Coef2_mean = Coef2_mean / num_batch
    Err = torch.sqrt( (Coef2_mean- Coef_mean**2)/Ns)
    return Coef_mean, Err


def flip2(samples, gate, k, site):
    """
    Given a sample state $a'$ and a 2qbit-gate, this method computes the associated states $a$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :return: A pair of tensors of sizes, (nb_samples * k**2, nb_qbits) and (nb_samples * k **2) respectively.
    """

    device = samples.device

    nb_samples = samples.size(0)
    nb_qbits = samples.size(1)
    # variable with the  flipped the measurements generated by a 2-qbit p_gate

    # repeat the samples K**2 sites. size = k**2 * nb_samples  x n_qbits
    flipped = samples.unsqueeze(1).repeat((1, k ** 2, 1)).view(k ** 2 * nb_samples, nb_qbits)
    a = torch.tensor(list(product(range(k), repeat=2)))  # possible combinations of outcomes on 2 qubits

    # replacing the outcomes with the possible flipped outcomes
    flipped[:, site[0]] = a[:, 0].repeat(nb_samples)
    flipped[:, site[1]] = a[:, 1].repeat(nb_samples)

    # getting the coefficients of the p-gates that accompany the flipped samples
    coef = gate[:, :, samples[:, site[0]], samples[:, site[1]]]

    # reshapes so that both coef and flipped have the same dim
    coef = coef.permute(2, 0, 1).contiguous().view(nb_samples * k ** 2)

    return flipped, coef


def flip2_reverse_presamples(samples, gate, k, site, model):
    """
    Given a sample state $a$ and a 2qbit-gate, this method computes the associated states $a'$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits) and (nb_samples) respectively.
    """

    device = samples.device

    nb_samples = samples.size(0)
    nb_qbits = samples.size(1)
    # variable with the  flipped the measurements generated by a 2-qbit p_gate

    # repeat the samples K**2 sites. size = k**2 * nb_samples  x n_qbits
    flipped = samples.unsqueeze(1).repeat((1, k ** 2, 1)).view(k ** 2 * nb_samples, nb_qbits)
    a = torch.tensor(list(product(range(k), repeat=2)))  # possible combinations of outcomes on 2 qubits

    # replacing the outcomes with the possible flipped outcomes
    flipped[:, site[0]] = a[:, 0].repeat(nb_samples)
    flipped[:, site[1]] = a[:, 1].repeat(nb_samples)

    # getting the coefficients of the p-gates that accompany the samples
    o_ab = gate[samples[:, site[0]], samples[:, site[1]], :, :]

    # reshapes so that both o_ab and flipped probability have the same dim
    o_ab = o_ab.view(nb_samples, k ** 2)

    # compute flipped probability
    pb = torch.exp(model.logP(flipped, look_ahead=True, device=device))
    pb = pb.view(nb_samples, k ** 2)
    # compute samples probability
    pa = torch.exp(model.logP(samples, look_ahead=True, device=device))
    # compute coef = sum {o_ab pb} / pa
    coef = torch.sum(o_ab * pb, dim=1) / pa


    return samples, coef


def flip2_reverse_core(samples, gate, k, site, model_copy, grad=False):
    """
    Given a sample state $a$ and a 2qbit-gate, this method computes the associated states $a'$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :param model_copy: copy of model that does not take gradient and before update
    :param grad: copy of model to take gradient or not
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits) and (nb_samples) respectively.
    """

    device = samples.device

    nb_samples = samples.size(0)
    nb_qbits = samples.size(1)
    # variable with the  flipped the measurements generated by a 2-qbit p_gate

    # repeat the samples K**2 sites. size = k**2 * nb_samples  x n_qbits
    flipped = samples.unsqueeze(1).repeat((1, k ** 2, 1)).view(k ** 2 * nb_samples, nb_qbits)
    a = torch.tensor(list(product(range(k), repeat=2)))  # possible combinations of outcomes on 2 qubits

    # replacing the outcomes with the possible flipped outcomes
    flipped[:, site[0]] = a[:, 0].repeat(nb_samples)
    flipped[:, site[1]] = a[:, 1].repeat(nb_samples)

    # getting the coefficients of the p-gates that accompany the samples
    o_ab = gate[samples[:, site[0]], samples[:, site[1]], :, :]

    # reshapes so that both o_ab and flipped probability have the same dim
    o_ab = o_ab.view(nb_samples, k ** 2)

    # compute flipped probability
    if grad:
        pb = torch.exp(sum_log_p(flipped, model_copy))
    else: 
        pb = torch.exp(model_copy(forward_type="logP", seq=flipped))
    pb = pb.view(nb_samples, k ** 2)
    p_new = torch.sum(o_ab * pb, dim=1)

    return p_new


def flip2_reverse(samples, gate, k, site, model, model_copy):
    """
    Given a sample state $a$ and a 2qbit-gate, this method computes the associated states $a'$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :param model: model that takes gradient
    :param model_copy: copy of model that does not take gradient and before update
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits) and (nb_samples) respectively.
    """

    coef = flip2_reverse_core(samples, gate, k, site, model_copy)
    # compute samples probability
    device = samples.device
    pa = torch.exp(model.logP(samples, look_ahead=True, device=device))
    # compute coef = sum {o_ab pb} / pa
    coef = coef / pa

    return samples, coef



def reverse_samples_lindblad(samples, operator, operator_b, k, model, grad=True):
    """
    Given a sample state $a$ and a 2qbit-gate, this method computes the associated states $a'$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param operator: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of the two-site lindblad operator.
    :param operator_b: the additional boundary operator
    :param k: number of measurements? (for our case it is 4)
    :param model: model that takes gradient
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits) and (nb_samples) respectively.
    """

    device = samples.device
    nb_samples = samples.size(0)
    nb_qbits = int(samples.size(1))

    p_new = flip2_reverse_core(samples, operator_b, k, [nb_qbits-1, nb_qbits-2], model, grad)

    for i in range(0, nb_qbits-1):
        p_new += flip2_reverse_core(samples, operator, k, [i, i+1], model, grad)

    return p_new


def reverse_samples_lindblad_pbc(samples, operator, k, model, grad=True):
    """
    Given a sample state $a$ and a 2qbit-gate, this method computes the associated states $a'$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param operator: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of the two-site lindblad operator.
    :param k: number of measurements? (for our case it is 4)
    :param model: model that takes gradient
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits) and (nb_samples) respectively.
    """

    device = samples.device
    nb_samples = samples.size(0)
    nb_qbits = int(samples.size(1))

    p_new = flip2_reverse_core(samples, operator, k, [0, 1], model, grad)

    for i in range(1, nb_qbits):
        p_new += flip2_reverse_core(samples, operator, k, [i%nb_qbits, (i+1)%nb_qbits], model, grad)
       

    return p_new

def point_to_number(r_num, c_num, num_columns):
    """
    covert the 2d point into a 1d number according to the sequence
    0 1 2
    3 4 5
    6 7 8
    :param r_num: the row number (first index) of the 2d point
    :param c_num: the column number (second index) of the 2d point
    :param num_columns: the total number of columns
    :return: the corresponding 1d number
    """
    return r_num * num_columns + c_num

def number_to_point(num, num_columns):
    """
    covert the 1d number into a 2d point according to the sequence
    0 1 2
    3 4 5
    6 7 8
    :param num: 1d number
    :param num_columns: the total number of columns
    :return: the corresponding 2d point in the order (row number, column number)
    """
    return num // num_columns, num % num_columns

def reverse_samples_lindblad_2d_pbc(samples, operator, k, model, num_rows, num_columns, grad=True):
    """
    Given a sample state $a$ and a 2qbit-gate, this method computes the associated states $a'$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param operator: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of the two-site lindblad operator.
    :param k: number of measurements? (for our case it is 4)
    :param model: model that takes gradient
    :param num_rows: the total number of rows
    :param num_columns: the total number of columns
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits) and (nb_samples) respectively.
    """

    device = samples.device
    nb_samples = samples.size(0)
    nb_qbits = int(samples.size(1))

    p_new = torch.zeros(nb_samples).to(device)

    for i in range(0, nb_qbits):
        r, c = number_to_point(i, num_columns)
        p_new += flip2_reverse_core(samples, operator, k, [i, point_to_number(r, (c+1)%num_columns, num_columns)], model, grad)
        p_new += flip2_reverse_core(samples, operator, k, [i, point_to_number((r+1)%num_rows, c, num_columns)], model, grad)
   

    return p_new






def sum_log_p(samples, model):
    """
    Returns the sum of the log probabilities.
    :param samples: Int tensor of size nb_samples x nb_qbits
    :param model: The model used to predict the likeness.
    :return: Sum of log probabilities.
    """
    return model('sum_log_p', samples)


def flip2_probs(samples, gate, k, site, model):
    """
    :param samples: Int tensor of size nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate, this gate should be the inverse gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :param model: The model used to predict the likeness.
    :return: sum O_ab p_NN_{b} of shape (nb_samples).
    """

    device = samples.device
    nb_samples = samples.size(0)
    nb_qbits = samples.size(1)
    nb_measurements = k

    # variable with the  flipped the measurements generated by a 2-qbit p_gate
    # repeat the samples K**2 sites. size = k**2 * nb_samples  x n_qbits
    flipped = samples.unsqueeze(1).repeat((1, k ** 2, 1)).view(k ** 2 * nb_samples, nb_qbits)
    a = torch.tensor(list(product(range(k), repeat=2)))  # possible combinations of outcomes on 2 qubits

    # replacing the outcomes with the possible flipped outcomes
    flipped[:, site[0]] = a[:, 0].repeat(nb_samples)
    flipped[:, site[1]] = a[:, 1].repeat(nb_samples)

    # getting the coefficients of the p-gates that accompany the samples
    o_ab = gate[samples[:, site[0]], samples[:, site[1]], :, :]
    # reshapes so that both o_ab and flipped probability have the same dim
    o_ab = o_ab.view(nb_samples, k ** 2)

    # Given a sample a_0, ..., a_n changes it to SOS, a_0, ..., a_{n-1}
    init = nb_measurements * torch.ones((nb_samples* k**2, 1), dtype=torch.long, device=device)
    input_sample = torch.cat([init, flipped], dim=1)[:, 0:nb_qbits]

    # The look_ahead mask is needed to only attend to previous qbits.
    probs = model(input_sample, look_ahead=True, device=device)  # n_samples x seq_len x nb_measurements
    log_p = torch.log(torch.softmax(probs, dim=2) + 1e-10)

    # Creates the one_hot_encodding
    eye = torch.eye(nb_measurements).to(device)
    one_hot = eye[flipped]

    pb = torch.exp((one_hot * log_p).sum(dim=1).sum(dim=1))
    pb = pb.view(nb_samples, k ** 2)
    p_new = torch.sum(o_ab * pb, dim=1)

    return p_new


def forward_backward_trapozoid_L1(samples, logP_samples, p_new1, p_new2, gamma=0):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param logP_samples: log probability of the given samples
    :param p_new1: the p_new corresponding to new model
    :param p_new2: the p_new corresponding to old model (do not require grad)
    :param gamma: weighted by the power of p_samples
    :return: sum abs(p_new1 - p_new2) / p_samples
    """
    p_samples = torch.exp(logP_samples)
    Loss = torch.abs(p_new1 - p_new2) * torch.pow(p_samples, gamma -1)
    return Loss.mean()