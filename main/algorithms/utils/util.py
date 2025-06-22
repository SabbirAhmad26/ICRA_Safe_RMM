import copy
import numpy as np
import math
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import KLDivLoss
from copy import deepcopy
from itertools import permutations, product

def mask_from_pooling(matrix, k):
    n, _ = matrix.shape
    
    # Initialize the boolean mask
    mask = np.zeros_like(matrix, dtype=bool)
    
    # Process each row
    for i in range(n):
        # Exclude the first column
        row = matrix[i, 1:]  # Last (n-1) values
        # Find indices of the k smallest values in the remaining row
        smallest_indices = np.argpartition(row, k)[:k]
        # Set the corresponding mask values to True
        mask[i, 1:][smallest_indices] = True
    
    return mask

def generate_boolean_mask(n, k):
    # Initialize a mask with all False
    mask = np.zeros((n, n), dtype=bool)
    
    # Randomly select k indices for each row to set to True
    for i in range(n):
        mask[i, np.random.choice(n, k, replace=False)] = True
    
    return mask

def generate_graph(num_cav, num_v, max_car, batch_size):
    cav_nodes = list(range(num_cav))
    ucv_nodes = list(range(num_cav, num_v))
    base_graph = list(permutations(cav_nodes, 2))
    base_graph += list(product(ucv_nodes, cav_nodes))
    base_graph = np.transpose(np.asarray(base_graph))
    if batch_size == 1:
        return base_graph

    edges = deepcopy(base_graph)
    for i in range(1, batch_size):
        edges = np.hstack((edges, base_graph+max_car*i))
    return edges

def generate_full_graph(num_agents):
    nodes = list(range(num_agents))
    edges = list(permutations(nodes, 2))
    edges = np.transpose(np.asarray(edges))
    return edges

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = th.from_numpy(input) if type(input) == np.ndarray else input
    return output

def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def param_schedule(curr_time, start_time, end_time, start_value, end_value, method='linear'):
    if curr_time <= start_time:
        return start_value
    elif curr_time >= end_time:
        return end_value
    
    ratio = (curr_time - start_time) / (end_time - start_time)
    delta = (end_value - start_value)
    if method == 'linear':
        return start_value + delta * ratio
    elif method == 'quad':
        return start_value + delta * (ratio ** 2)
    elif method == 'exp':
        return start_value + delta * (ratio ** 3)
    else:
        return NotImplementedError


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape

def get_num_actions_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        num_actions = act_space.n
    elif act_space.__class__.__name__ == "MultiDiscrete":
        #act_shape = act_space.shape
        raise NotImplementedError
    elif act_space.__class__.__name__ == "Box":
        #act_shape = act_space.shape[0]
        raise NotImplementedError
    elif act_space.__class__.__name__ == "MultiBinary":
        #act_shape = act_space.shape[0]
        raise NotImplementedError
    else:  # agar
        #act_shape = act_space[0].shape[0] + 1  
        raise NotImplementedError
    return num_actions


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c


'''
    rewards: num_steps * num_agents numpy array
    gamma: discount factor
'''
def compute_discounted_return(rewards, gamma):
    n_steps, n_agents = rewards.shape[0], rewards.shape[1]
    discounted = np.zeros(n_agents)
    for step in reversed(range(n_steps)):
        discounted = gamma * discounted + rewards[step]
    return discounted

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

#####

def identity(x):
    return x

def entropy(p):
    return -th.sum(p * th.log(th.max(p, th.tensor(1e-10))), 1)

def min_clip(t, threshold=1e-10):
    return th.max(t, th.tensor(threshold))

def kl_log_probs(log_p1, log_p2):
    return -th.sum(th.exp(log_p1)*(log_p2 - log_p1), 1)


def index_to_one_hot(index, dim):
    if isinstance(index, np.int32) or isinstance(index, np.int64) or isinstance(index, int):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.
    return one_hot


def to_tensor_var(x, use_cuda=True, dtype="float"):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))


def agg_double_list(l):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    #print("print eval score", np.array(l).shape)
    s = [np.sum(np.array(l_i), 0) for l_i in l] # k episodes, 
    # for each episodes, an array of n agents' total scores
    #print('s', s)
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std

def get_running_reward(reward_array: np.ndarray, window=10):
    """calculate the running reward, i.e. average of last `window` elements from rewards"""
    running_reward = np.zeros_like(reward_array)
    window = min(window, reward_array.shape[0])
    for i in range(window - 1):
        running_reward[i] = np.mean(reward_array[:i + 1])
    for i in range(window - 1, len(reward_array)):
        running_reward[i] = np.mean(reward_array[i - window + 1:i + 1])
    return running_reward


def shape_equal_cmp(*args):
    '''
    Checks that the shapes of the passed arguments are equal
    Inputs:
    - All arguments should be tensors
    Returns:
    - True if all arguments have the same shape, else ValueError
    '''
    for i in range(len(args)-1):
        if args[i].shape != args[i+1].shape:
            s = "\n".join([str(x.shape) for x in args])
            raise ValueError("Expected equal shapes. Got:\n%s" % s)
    return True


"""Computing an estimated upper bound of KL divergence using SGLD."""
def get_state_kl_bound_sgld_disc(net, batch_states, eps=2, steps=10, rnn_states=None, masks=None):
    _, _, _, old_action_probs = net(batch_states, rnn_states, masks)
    old_action_probs = old_action_probs.detach()
    # _, old_action_log_probs, _ = net(batch_states).detach()

    # upper and lower bounds for clipping
    states_ub = batch_states + eps
    states_lb = batch_states - eps
    step_eps = eps / steps
    # SGLD noise factor. We set (inverse) beta=1e-5 as gradients are relatively small here.
    beta = 1e-5
    noise_factor = np.sqrt(2 * step_eps * beta)
    noise = th.randn_like(batch_states) * noise_factor
    var_states = (batch_states.clone() + noise.sign() * step_eps).detach().requires_grad_()
    
    kl_loss = KLDivLoss(reduction='none', log_target=False)
    for i in range(steps):
        # Find a nearby state new_phi that maximize the difference
        _, _, _, temp_action_probs = net(var_states, rnn_states, masks)
        #print("TEMP: ", temp_action_probs.shape)
        non_zero_bound = th.full_like(temp_action_probs, 1e-10, requires_grad=True)
        log_bounded_temp_action_probs = th.maximum(temp_action_probs, non_zero_bound).log()
        kl = kl_loss(log_bounded_temp_action_probs, old_action_probs)
        kl = kl.sum(dim=-1, keepdim=True).mean()
        # Need to clear gradients before the backward() for policy_loss
        kl.backward()
        # Reduce noise at every step.
        noise_factor = np.sqrt(2 * step_eps * beta) / (i+2)
        # Project noisy gradient to step boundary.
        update = (var_states.grad + noise_factor * th.randn_like(var_states)).sign() * step_eps
        var_states.data += update
        # clip into the upper and lower bounds
        var_states = th.maximum(var_states, states_lb)
        var_states = th.minimum(var_states, states_ub)
        var_states = var_states.detach().requires_grad_()
    net.zero_grad()
    _, _, _, final_action_probs = net(var_states.requires_grad_(False), rnn_states, masks)
    non_zero_bound = th.full_like(final_action_probs, 1e-10, requires_grad=True)
    return kl_loss(th.maximum(final_action_probs, non_zero_bound).log(), old_action_probs).sum(dim=-1, keepdim=True)


if __name__ == "__main__":
    print(generate_graph(2, 4, 5, 3))
