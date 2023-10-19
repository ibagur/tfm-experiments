import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .quant_layer import Conv2d_Q, Linear_Q
from .ppo_blip_utils import update_fisher_exact

import logging #CHECK

class PPO_BLIP_EWC():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 ewc_epoch = 1,
                 ewc_lambda = 5000,
                 online = False,
                 optimizer='Adam', 
                 update_bits_rounding='floor', 
                 fisher_term='f0t'
                 ):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.ewc_epoch = 1
        self.ewc_lambda = ewc_lambda

        self.lr = lr
        self.eps = eps
        self.EWC_task_count = 0
        self.task_count = 0
        self.divide_factor = 0
        self.online = online

        self.optimizer_type = optimizer

        # new parameters to select different sup-approaches
        self.update_bits_rounding = update_bits_rounding
        self.fisher_term = fisher_term

        # to log variables content #CHECK
        logging.basicConfig(filename='reg_loss.log', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def renew_optimizer(self):
        #self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=self.eps)
        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.lr, eps=self.eps)
        elif self.optimizer_type == 'RMSProp':
            self.optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.lr, eps=self.eps)
        else:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.lr, eps=self.eps) 
            
    def update(self, rollouts, task_num):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch, task_num)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                # add EWC or SPP loss
                reg_loss = self.ewc_lambda * self.blip_ewc_loss(task_num)

                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef + reg_loss).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                # clipping parameters accordingly
                for m in self.actor_critic.features.modules():
                    if isinstance(m, Conv2d_Q) or isinstance(m, Linear_Q):
                        m.clipping()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def ng_post_processing(self, rollouts, task_id):
        self.estimate_fisher(rollouts, task_id)
        for m in self.actor_critic.features.modules():
            if isinstance(m, Conv2d_Q) or isinstance(m, Linear_Q):
                if self.update_bits_rounding == 'floor':
                    m.update_bits_floor(task=task_id, C=0.5/math.log(2))
                elif self.update_bits_rounding == 'ceil':
                    m.update_bits(task=task_id, C=0.5/math.log(2))
                m.sync_weight()
                m.update_fisher(task=task_id)

    def estimate_fisher(self, rollouts, task_num):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        def _save_state(module, input, results):        
            module._state = input[0].clone()

        def _save_costate(module, grad_input, grad_output):
            module._costate = grad_output[0].clone()

        # register hooks
        for m in self.actor_critic.features.modules():
            if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                m.handle_forward = m.register_forward_hook(_save_state)
                m.handle_backward = m.register_backward_hook(_save_costate)
                #m.handle_backward = m.register_full_backward_hook(_save_costate)

        self.actor_critic.eval()
        
        total_data = 0
        num_round = 1
        for _ in range(num_round):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, 32)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, 32)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, _, \
                   _, _, masks_batch, _, _ = sample

                batch_size_t = obs_batch.shape[0]
                total_data += batch_size_t

                # clear gradient
                self.actor_critic.zero_grad()

                # get action distribution
                actor_features, _ = self.actor_critic.features(obs_batch, 
                    recurrent_hidden_states_batch, masks_batch)
                batch_action_dist = self.actor_critic.dist[task_num](actor_features)
                sampled_actions = batch_action_dist.sample()
                sampled_action_log_probs = batch_action_dist.log_probs(sampled_actions)
                (-sampled_action_log_probs.mean()).backward()

                update_fisher_exact(self.actor_critic)

                self.actor_critic.zero_grad()

        for m in self.actor_critic.features.modules():
            if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                m.Fisher_w /= total_data
                if m.bias is not None:
                    m.Fisher_b /= total_data
                m.handle_forward.remove()
                m.handle_backward.remove()
        self.actor_critic.train()

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.task_count = 1 if self.online else self.task_count + 1
        # self.divide_factor += 1

    def blip_ewc_loss(self, task):
        if self.task_count > 0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for m in self.actor_critic.features.modules():
                if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                    diff_w = m.weight - m.prev_weight
                    #TEST use F_0_t
                    if self.fisher_term == 'f0t':
                        fisher_w = self.Fisher_w_0_current
                    elif self.fisher_term == 'ft':
                        #TEST using just F_t
                        fisher_w = m.Fisher_w_current
                    losses.append((fisher_w * (diff_w) ** 2).sum())

                    if m.bias is not None:
                        diff_b = m.bias - m.prev_bias
                        if self.fisher_term == 'f0t':
                            fisher_b = self.Fisher_b_0_current
                        elif self.fisher_term == 'ft':
                            #TEST using just F_t
                            fisher_b = m.Fisher_b_current
                        losses.append((fisher_b * (diff_b) ** 2).sum())

                    # if self.flag_log == True: #CHECK
                    #     # Logging the details
                    #     self.logger.info(f"m.weight: {m.weight}")
                    #     self.logger.info(f"m.prev_weight: {m.prev_weight}")
                    #     self.logger.info(f"m.Fisher_w_current: {m.Fisher_w_current}")
                    #     self.logger.info(f"m.Fisher_w_prev: {m.Fisher_w_prev}")
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1. / 2) * sum(losses)            
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return 0.
