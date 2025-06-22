import torch
import torch.nn as nn
import torch.nn.functional as F

def initial_bounds(x0, epsilon):
    '''
    x0 = input, b x c x h x w
    '''
    upper = x0+epsilon
    lower = x0-epsilon
    return upper, lower

def weighted_bound(layer, prev_upper, prev_lower):
    prev_mu = (prev_upper + prev_lower)/2
    prev_r = (prev_upper - prev_lower)/2
    mu = layer(prev_mu)
    if type(layer)==nn.Linear:
        r = F.linear(prev_r, torch.abs(layer.weight))
    elif type(layer)==nn.Conv2d:
        r = F.conv2d(prev_r, torch.abs(layer.weight), stride=layer.stride, padding=layer.padding)
    
    upper = mu + r
    lower = mu - r
    return upper, lower

def activation_bound(layer, prev_upper, prev_lower):
    upper = layer(prev_upper)
    lower = layer(prev_lower)
    return upper, lower

def network_bounds(model, x0, epsilon):
    '''
    get inteval bound progation upper and lower bounds for the actiavtion of a model
    
    model: a nn.Sequential module
    x0: input, b x input_shape
    epsilon: float, the linf distance bound is calculated over
    '''
    upper, lower = initial_bounds(x0, epsilon)
    for layer in model.modules():
        if type(layer) in (nn.Sequential,):
            pass
        elif type(layer) in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.Flatten):
            upper, lower = activation_bound(layer, upper, lower)
        elif type(layer) in (nn.Linear, nn.Conv2d):
            upper, lower = weighted_bound(layer, upper, lower)
        # else:
            # print('Unsupported layer:', type(layer))
    return upper, lower

def worst_action_select(worst_q, upper_q, lower_q, use_cuda=False):
    if use_cuda:
        mask = torch.zeros(upper_q.size()).cuda()
    else:
        mask = torch.zeros(upper_q.size()).cpu()
    for i in range(upper_q.size()[1]):
        upper = upper_q[:, i].view(upper_q.size()[0], 1)
        if_perturb = (upper > lower_q).all(1) # this uq of action is better than all lq of all actions
        # mask==0 i.e. exist an action's lq that is >=  current action's uq
        mask[:, i] = if_perturb.byte()

    worst_q = worst_q.masked_fill_(mask==0, 1e9)
    worst_actions = worst_q.min(1)[-1].unsqueeze(1)

    return worst_actions


def worst_action_from_all(worst_q):
    worst_actions = worst_q.min(1)[-1].unsqueeze(1)
    return worst_actions

'''
Target action: using gradient descent to find the worst action with worst worst-case q value
used in continuous actions
'''
def worst_action_pgd(q_net, policy_net, states, eps=0.0005, maxiter=100):
    with torch.no_grad():
        action_ub, action_lb = network_bounds(policy_net, states, eps)
        action_means, _ = policy_net(states)
    # print(action_means)
    # var_actions = Variable(action_means.clone().to(device), requires_grad=True)
    var_actions = action_means.requires_grad_()
    step_eps = (action_ub - action_lb) / maxiter

    for i in range(maxiter):
        worst_q = q_net(torch.cat((states, var_actions), dim=1))
        worst_q.backward(torch.ones_like(worst_q))
        grad = var_actions.grad.data  
        var_actions.data -= step_eps * torch.sign(grad)
        var_actions = torch.max(var_actions, action_lb)
        var_actions = torch.min(var_actions, action_ub)
        var_actions = var_actions.detach().requires_grad_() 
    q_net.zero_grad()
    return var_actions.detach()