import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR #TEST scheduler

from .quant_layer import Conv2d_Q, Linear_Q
from .ppo_blip_utils import update_fisher_exact
#import logging


class PPO_BLIP_SPP_MASK():
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
                 spp_lambda=4,
                 initial_prune_percent = 30.0,
                 prune_percent_decay = 0.8,
                 num_tasks = 3,
                 use_scheduler = False,
                 prune_higher = False
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
        self.spp_lambda = spp_lambda
        self.omega_w = None
        self.omega_b = None

        self.initial_prune_percent = initial_prune_percent
        self.prune_percent_decay = prune_percent_decay
        self.prune_percent = 0.0


        # For SPP Mask optimizer renew
        self.value_loss = 0.0
        self.action_loss = 0.0
        self.dist_entropy = 0.0
        self.num_tasks = num_tasks
        self.mask_pre_w = None
        self.mask_pre_b = None
        self.use_scheduler = use_scheduler
        self.prune_higher = prune_higher

        # to log variables content
        #logging.basicConfig(filename='reg_loss.log', level=logging.INFO)
        #self.logger = logging.getLogger(__name__)

    def renew_optimizer(self):
        #self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=self.eps)
        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.lr, eps=self.eps)
        elif self.optimizer_type == 'RMSProp':
            self.optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.lr, eps=self.eps)
        else:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.lr, eps=self.eps) 

    # for blip_spp_mask, to initialize the optimizer
    def renew_optimizer_mask(self, task_num):

        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.lr, eps=self.eps)
        elif self.optimizer_type == 'RMSProp':
            self.optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.lr, eps=self.eps)
        else:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.lr, eps=self.eps) 

        if task_num == 0:
            # Define the learning rate scheduler. We use StepLR which multiplies the learning rate by gamma every 'epoches' steps
            if self.use_scheduler == True:
                self.scheduler = StepLR(self.optimizer, step_size=self.ppo_epoch, gamma=0.99)
        else:

            if self.initial_prune_percent == 0:
                mask_w_ = torch.ones_like(self.omega_w)
                mask_b_ = torch.ones_like(self.omega_b)
            else:
                mask_w_ = self.mask_pre_w
                mask_b_ = self.mask_pre_b
                self.apply_prune_weights(mask_w_, mask_b_)
        
            # Define the learning rate scheduler. We use StepLR which multiplies the learning rate by gamma every 'epoches' steps
            if self.use_scheduler == True:
                self.scheduler = StepLR(self.optimizer, step_size=self.ppo_epoch, gamma=0.99)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Backward pass to get the gradients
            # add SPP loss
            reg_loss = self.spp_lambda * self.blip_spp_loss(task_num)

            (self.value_loss * self.value_loss_coef + self.action_loss - 
             self.dist_entropy * self.entropy_coef + reg_loss).backward(retain_graph=True)

            # Modify gradients using masks
            with torch.no_grad():
                w_counter = 0  # Counter for modules with weights
                b_counter = 0  # Counter for modules with biases
                for m in self.actor_critic.features.modules():
                    if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                        m.weight.grad.data.mul_(mask_w_[0][w_counter])
                        w_counter += 1
                        if m.bias is not None:
                            m.bias.grad.data.mul_(mask_b_[0][b_counter])
                            b_counter += 1
            
            # Step the optimizer
            self.optimizer.step()
            # Update learning rate
            if self.use_scheduler == True:
                self.scheduler.step()

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

                # add SPP loss
                # reg_loss = self.spp_lambda * self.blip_spp_loss(task_num)

                # (value_loss * self.value_loss_coef + action_loss -
                #  dist_entropy * self.entropy_coef + reg_loss).backward()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
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

        if self.use_scheduler == True:
                self.scheduler.step()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        self.value_loss = value_loss_epoch
        self.action_loss = action_loss_epoch
        self.dist_entropy = dist_entropy_epoch

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

    def update_omega(self, rollouts, task_num):

        avg_omega_w, avg_omega_b = self.compute_avg_omega(rollouts, task_num)

        if task_num == 0:
            self.omega_w, self.omega_b = avg_omega_w, avg_omega_b
        else:
            self.omega_w = torch.cat((self.omega_w, self.omega_w+avg_omega_w), dim=0)
            self.omega_b = torch.cat((self.omega_b, self.omega_b+avg_omega_b), dim=0)

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.task_count = 1 if self.online else self.task_count + 1

    def compute_avg_omega(self, rollouts, task_num):

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)
        
        self.actor_critic.eval()

        total_data = 0

        if self.actor_critic.is_recurrent:
            data_generator = rollouts.recurrent_generator(
                advantages, 32)
        else:
            data_generator = rollouts.feed_forward_generator(
                advantages, 32)
        cumulative_omega_w = None  # Variable to hold cumulative omega
        cumulative_omega_b = None  # Variable to hold cumulative omega
        batch_num = 0  # Variable to keep track of the batch number

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
            (-sampled_action_log_probs.mean()).backward(retain_graph=True)

            #param_im_w, param_im_w = self.cal_importance(sampled_action_log_probs)

            omega_w, omega_b = self.compute_omega(obs_batch, sampled_action_log_probs, batch_size_t)

            if cumulative_omega_w is None:
                cumulative_omega_w = omega_w
                cumulative_omega_b = omega_b
            else:
                cumulative_omega_w += omega_w  # Summing omega_m values across batches
                cumulative_omega_b += omega_b
            batch_num += 1

        avg_omega_w = (cumulative_omega_w / batch_num).view(1,2)
        avg_omega_b = (cumulative_omega_b / batch_num).view(1,2)

        self.actor_critic.train()

        return avg_omega_w, avg_omega_b

    def compute_omega(self, obs_batch, log_probs_batch, batch_size):
        omega_w = []
        omega_b = []
        batch_num = math.ceil(len(obs_batch) / batch_size)
        
        # Split the log_probs_batch into smaller batches
        log_probs_batches = torch.split(log_probs_batch, batch_size, dim=0)

        for log_probs in log_probs_batches:
            param_im_w, param_im_b = self.cal_importance(log_probs)
            omega_w.append(param_im_w)
            omega_b.append(param_im_b)
        
        omega_w = [sum(tensors) / batch_num for tensors in zip(*omega_w)]
        omega_b = [sum(tensors) / batch_num for tensors in zip(*omega_b)]

        om_w = torch.stack(omega_w)
        om_b = torch.stack(omega_b)
        
        # Flatten the list
        #omega_w = [item for sublist in omega_w for item in sublist]
        #omega_b = [item for sublist in omega_b for item in sublist]

        #omega_w = torch.stack(omega_w).sum(dim=0) / batch_num
        #omega_b = torch.stack(omega_b).sum(dim=0) / batch_num

        return om_w, om_b
    
    def cal_importance(self, log_probs):

        importance_w = []
        importance_b = []

        for m in self.actor_critic.features.modules():
            if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                I = -torch.sum(log_probs.exp() * log_probs)
                with torch.no_grad():
                    # Compute the gradients of the information entropy with respect to the current weights
                    grads_w = torch.autograd.grad(I, m.weight, retain_graph=True)[0]
                    # Calculate the importance based on the given formula
                    term1_w = grads_w * (m.weight - m.prev_weight)
                    term2_w = 0.5 * (m.weight - m.prev_weight).pow(2) * grads_w.pow(2)
                    imp_w = torch.max(term1_w + term2_w, torch.tensor(0.0))
                    importance_w.append(torch.mean(imp_w))

                    if m.bias is not None:
                        # Compute the gradients of the information entropy with respect to the current biases                
                        grads_b = torch.autograd.grad(I, m.bias, retain_graph=True)[0]
                        # Calculate the importance based on the given formula
                        term1_b = grads_b * (m.bias - m.prev_bias)
                        term2_b = 0.5 * (m.bias - m.prev_bias).pow(2) * grads_b.pow(2)
                        imp_b = torch.max(term1_b + term2_b, torch.tensor(0.0))
                        importance_b.append(torch.mean(imp_b))
        return importance_w, importance_b
    
    def update_prune_mask(self, rollouts, task_num):

        if task_num == 0:
            self.prune_percent = self.initial_prune_percent
        else:
            self.prune_percent = int(self.prune_percent * self.prune_percent_decay)
        
        if self.prune_percent != 0:
            silence_w, silence_b = self.compute_avg_omega(rollouts, task_num)

            if task_num == 0:
                mask_w, mask_b = self.get_mask_matrix1(silence_w, silence_b, self.prune_percent)
            elif task_num < (self.num_tasks - 1):
                mask_w, mask_b = self.get_mask_matrix2(self.mask_pre_w, self.mask_pre_b, silence_w, silence_b, self.prune_percent)
            else:
                mask_w, mask_b = self.mask_pre_w, self.mask_pre_b

            self.mask_pre_w = mask_w
            self.mask_pre_b = mask_b

    # internal method
    def _apply_mask1(self, omega, percent):
        mask = torch.ones_like(omega)
        # Calculate the percentile threshold
        threshold = torch.quantile(omega.float(), percent / 100.0)
        # Zero out values below the threshold
        if self.prune_higher == True: #TEST prune higher importance values (make mask = 0)
            mask[omega >= threshold] = 0
        else:
            mask[omega < threshold] = 0
        
        return mask

    # Input here is the avg_omega
    def get_mask_matrix1(self, omega_w, omega_b, percent):
        mask_w = self._apply_mask1(omega_w, percent)
        mask_b = self._apply_mask1(omega_b, percent)
        return mask_w, mask_b         

    # internal method
    def _apply_mask2(self, mask_pre, silence, step_percent):
        mask = torch.ones_like(mask_pre)
        # Filter the values in silence where mask_pre is zero
        s2 = silence[mask_pre == 0]
        # Check if s2 is empty, if so, set threshold to infinity
        if s2.nelement() == 0:
            threshold_current = torch.tensor(float('inf'))
        else:
            # Calculate the step_percent percentile value of s2
            threshold_current = torch.quantile(s2.float(), step_percent / 100.0)
        # Set values in m_a to zero based on the conditions
        if self.prune_higher == True:  #TEST prune higher importance values (make mask = 0)
            mask[(mask_pre == 0) & (silence >= threshold_current)] = 0
        else:
            mask[(mask_pre == 0) & (silence < threshold_current)] = 0

        return mask
    
    def get_mask_matrix2(self, mask_pre_w, mask_pre_b, silence_w, silence_b, step_percent):
        mask_w = self._apply_mask2(mask_pre_w, silence_w, step_percent)
        mask_b = self._apply_mask2(mask_pre_b, silence_b, step_percent)
        return mask_w, mask_b
    
    def apply_prune_weights(self, mask_w, mask_b):
        w_counter = 0  # Counter for modules with weights
        b_counter = 0  # Counter for modules with biases
        
        for m in self.actor_critic.features.modules():
            if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                m.weight.data.mul_(mask_w[0][w_counter])
                w_counter += 1

                if m.bias is not None:
                    m.bias.data.mul_(mask_b[0][b_counter])
                    b_counter += 1


    def blip_spp_loss(self, task_num):

        if self.task_count > 0:
            losses = []
            w_counter = 0  # Counter for modules with weights
            b_counter = 0  # Counter for modules with biases
            for m in self.actor_critic.features.modules():
                if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                    diff_w = m.weight - m.old_weight
                    # use task_num-1 because self.omega_w[task_num] will only exist after task is fully completed
                    losses.append((self.omega_w[task_num-1][w_counter] * (diff_w)**2).sum())
                    w_counter += 1  # Increment weight counter

                    if m.bias is not None:
                        diff_b = m.bias - m.old_bias
                        losses.append((self.omega_b[task_num-1][b_counter] * (diff_b)**2).sum())
                        b_counter += 1  # Increment bias counter if bias exists

            return sum(losses)
        else:
            return 0.
