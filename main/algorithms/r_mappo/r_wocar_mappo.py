import numpy as np
import math
import torch
import torch.nn as nn
from main.algorithms.utils.valuenorm import ValueNorm
from main.algorithms.utils.util import check, get_gard_norm, huber_loss, mse_loss, \
    soft_update, hard_update, index_to_one_hot, get_state_kl_bound_sgld_disc
from main.algorithms.utils.eps_scheduler import LinearSchduler_Epoch
from main.algorithms.utils.ibp import worst_action_from_all

class R_WOCAR_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        if args.env_name == 'CARLA':
            self.discrete_action_space = args.discretize + 3
        elif args.env_name == 'MPE':
            if args.scenario_name == 'simple_spread':
                self.discrete_action_space = 5
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        # wocar
        self.worst_q_weight = args.wqw
        self.weight_schedule = args.weight_schedule
        self.robust_ppo_reg = args.robust_reg
        self.wq_loss_coef = args.wq_loss_coef
        self.robust_eps_scheduler = LinearSchduler_Epoch(args.eps, args.eps_start, 1.)
        self.q_weight_scheduler = LinearSchduler_Epoch(self.worst_q_weight, args.eps_start, 1.)
        self.tau = 0.01
        self.q_gamma = 0.99

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_centralized_V = args.use_centralized_V
        self.num_agents = args.num_agents

        if self._use_centralized_V:
            assert args.share_policy
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def cal_worst_q_loss(self, obs_batch, masks_batch, one_hot_action_batch, \
                         next_obs_batch, next_masks_batch, reward_batch):
        current_Q_values = self.policy.evaluate_worst_q_values(obs_batch, masks_batch, one_hot_action_batch)
        next_worst_q = self.policy.get_worst_q_values(next_obs_batch, next_masks_batch, target_q=True)
        worst_actions = worst_action_from_all(next_worst_q)
        next_worst_q = next_worst_q.gather(1, worst_actions)
        target_Q_values = reward_batch + self.q_gamma * next_worst_q
        assert(next_worst_q.shape==target_Q_values.shape)
        #loss = F.smooth_l1_loss(current_Q_values, target_Q_values)
        if self._use_huber_loss:
            worst_q_loss = huber_loss(target_Q_values-current_Q_values, self.huber_delta)
        else:
            worst_q_loss = mse_loss(target_Q_values-current_Q_values)

        worst_q_loss = (worst_q_loss * masks_batch).sum() / masks_batch.sum()
        return worst_q_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, \
        next_obs_batch, next_masks_batch, reward_batch = sample

        one_hot_action_batch = np.vstack([index_to_one_hot(int(action), self.discrete_action_space) for action in actions_batch])
        #print("actions", actions_batch.shape)
        #print("share_obs", share_obs_batch.shape)
        #print("obs", obs_batch.shape)
        #print("mask", masks_batch.shape)
        #print("active mask", active_masks_batch.shape)
        #print("advantage", adv_targ.shape)
        #print("next obs", next_share_obs_batch.shape)
        #print("next mask", next_masks_batch.shape)
        #print("reward", reward_batch.shape, type(reward_batch))
        #print("onehot", one_hot_action_batch.shape, type(one_hot_action_batch))
        '''
        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        next_obs_batch = check(next_obs_batch).to(**self.tpdv)
        next_masks_batch = check(next_masks_batch).to(**self.tpdv)
        reward_batch = check(reward_batch).to(**self.tpdv)
        one_hot_action_batch = check(one_hot_action_batch).to(**self.tpdv)
        rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        '''

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)
        reward_batch = check(reward_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        
        # add worst_q term in actor loss
        worst_qs = self.policy.evaluate_worst_q_values(obs_batch, masks_batch, one_hot_action_batch)

        q_weight = self.q_weight_scheduler.get_eps()
        if self.weight_schedule == 'constant':
            q_weight = self.worst_q_weight
        elif self.weight_schedule == 'linear':
            pass
        elif self.weight_schedule == 'exp':
            q_weight = math.pow(q_weight, 3)
        else:
            raise NotImplementedError("Unsupported weight schedule")
        if q_weight > 1:
            q_weight = 1

        adv_targ += q_weight * worst_qs
        #kl_upper_bound = get_state_kl_bound_sgld_disc(self.policy.actor, obs_batch, \
        #    eps=0.2, steps=5, rnn_states=rnn_states_batch, masks=masks_batch).mean()

        
        kl_upper_bound = get_state_kl_bound_sgld_disc(self.policy.actor, obs_batch, \
                            eps=self.robust_eps_scheduler.get_eps(), steps=20, \
                            rnn_states=rnn_states_batch, masks=masks_batch).mean()
        '''
        kl_upper_bound = get_state_kl_bound_sgld_disc(self.policy.actor, obs_batch, \
                            eps=10., steps=10, \
                            rnn_states=rnn_states_batch, masks=masks_batch)
        '''
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        #policy_loss = policy_action_loss
        #policy_loss = policy_action_loss + self.robust_ppo_reg * kl_upper_bound
        reg_loss = ((values.detach() - worst_qs.detach()) * kl_upper_bound).mean()
        policy_loss = policy_action_loss + self.robust_ppo_reg * reg_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            #(policy_loss - dist_entropy * self.entropy_coef).backward()
            policy_loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        # worst_q_update
        worst_q_loss = self.cal_worst_q_loss(obs_batch, masks_batch, one_hot_action_batch, \
                         next_obs_batch, next_masks_batch, reward_batch)

        self.policy.worst_q_optimizer.zero_grad()
        (worst_q_loss * self.wq_loss_coef).backward()
        if self._use_max_grad_norm:
            wq_grad_norm = nn.utils.clip_grad_norm_(self.policy.worst_Q.parameters(), self.max_grad_norm)
        else:
            wq_grad_norm = get_gard_norm(self.policy.worst_Q.parameters())
        
        self.policy.worst_q_optimizer.step()
        soft_update(self.policy.target_worst_Q, self.policy.worst_Q, self.tau)
        
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, \
                worst_qs, reg_loss, worst_q_loss, wq_grad_norm

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_centralized_V:
            # A_t^i = r_t^i + \frac{1}{N}(Return_t - V_t - r_t)
            if self._use_popart or self._use_valuenorm:
                advantages = buffer.rewards + (buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1]) - buffer.total_rewards) / self.num_agents 
            else:
                advantages = buffer.rewards + (buffer.returns[:-1] - buffer.value_preds[:-1] - buffer.total_rewards) / self.num_agents
        else:
            if self._use_popart or self._use_valuenorm:
                advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
            else:
                advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['worst_qs'] = 0
        train_info['VB_state_regularization'] = 0
        train_info['worst_q_loss'] = 0
        train_info['worst_q_grad_norm'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, \
                worst_qs, reg_loss, worst_q_loss, wq_grad_norm \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['worst_qs'] += worst_qs.mean().item()
                train_info['VB_state_regularization'] += reg_loss.item()
                train_info['worst_q_loss'] += worst_q_loss.item()
                train_info['worst_q_grad_norm'] += wq_grad_norm

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.worst_Q.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.worst_Q.eval()
