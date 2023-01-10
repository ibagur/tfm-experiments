import numpy
import torch
import torch.nn.functional as F
# import from base_pc
from torch_ac.algos.base_pc import BaseAlgoPC


class PPOPCAlgo(BaseAlgoPC):
    """The Policy Consolidation for Continual Reinforcement Learning algorithm
    ([Kaplanis et al., 2019](https://arxiv.org/abs/1902.00255v2))."""

    def __init__(self, envs, acmodels, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 cascade_depth=1,
                 flow_factor=1.0,
                 mesh_factor = 4.0,
                 imp_sampling = 'normal',
                 imp_clips=[-5,5],
                 dynamic_neglogpacs=False,
                 lrs = 0.001,
                 lrs_fn = 0.001,
                 clipranges = 0.2,
                 optimizer_type='adam',
                 scheduler_flag=None,                 
                 reshape_reward=None, 
                 ):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodels, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, cascade_depth,
                         reshape_reward
                         )

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim_eps = adam_eps

        assert self.batch_size % self.recurrence == 0

        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.acmodels[0].parameters(), lr=lrs[0], eps=self.optim_eps)
        elif optimizer_type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.acmodels[0].parameters(), lr=lrs[0], eps=self.optim_eps)
        for i in range(1,self.cascade_depth):
            self.optimizer.add_param_group({'params': [*self.acmodels[i].parameters()], 'lr':lrs[i], 'eps':self.optim_eps})
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        # create cascade of optimizers and schedulers for each policy
        # optimizers = []
        # schedulers = []
        # for i in range(self.cascade_depth):
        #     if optimizer_type == 'adam':
        #         optimizer = torch.optim.Adam(self.acmodels[i].parameters(), lrs[i], eps=self.optim_eps)
        #     elif optimizer_type == 'rmsprop':
        #         optimizer = torch.optim.RMSprop(self.acmodels[i].parameters(), lrs[i], eps=self.optim_eps)
        #     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        #     optimizers.append(optimizer)
        #     schedulers.append(scheduler)
        # self.optimizers = optimizers
        # self.schedulers = schedulers

        self.batch_num = 0

        self.flow_factor = flow_factor
        self.mesh_factor = mesh_factor
        self.imp_sampling = imp_sampling
        self.imp_clips = imp_clips
        self.dynamic_neglogpacs = dynamic_neglogpacs
        self.lrs = lrs
        self.lrs_fn = lrs_fn # cascade policies lrs decay function
        self.clipranges = torch.tensor(clipranges)
        self.scheduler_flag = scheduler_flag

    def update_parameters_cascade(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodels[0].recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]
                    log_probs_cascade = []

                    # Transpose cascade variables
                    #sb.actions_cascade = torch.t(sb.actions_cascade)
                    sb.values_cascade = torch.t(sb.values_cascade)
                    sb.log_probs_cascade = torch.t(sb.log_probs_cascade)
                    sb.cumlog_probs_cascade = torch.t(sb.cumlog_probs_cascade)

                    for j in range(self.cascade_depth):
                        # Compute loss
                        #TODO Review for recurrent
                        if self.acmodels[0].recurrent:
                            dist, val, memory = self.acmodels[j](sb.obs, memory * sb.mask)
                        else:
                            dist, val = self.acmodels[j](sb.obs)
                        # Get the logprob of the visible policy actions using each policy distribution
                        log_probs_cascade.append(dist.log_prob(sb.action))
                        #log_probs_cascade.append(dist.log_prob(sb.actions_cascade[j]))
                        if j == 0:
                            value = val
                            entropy = dist.entropy().mean()

                    # Calculate importance factors cascade for clipped-PPO
                    
                    imp_factors = []
                    # not full-kl
                    if self.imp_sampling == 'normal':
                        for k in range(self.cascade_depth):
                            imp_factor = torch.exp(sb.cumlog_probs_cascade[0] - sb.cumlog_probs_cascade[k])
                            imp_factors.append(imp_factor)
                    elif self.imp_sampling == 'clipped':
                        assert len(self.imp_clips)==2
                        for k in range(self.cascade_depth):
                            imp_factor = torch.exp(torch.clamp(sb.cumlog_probs_cascade[0] - sb.cumlog_probs_cascade[k],self.imp_clips[0], self.imp_clips[1]))
                            imp_factors.append(imp_factor)
                    elif self.imp_sampling =='none':
                        imp_factors = [torch.ones(1) for i in range(self.cascade_depth)]

                    # Clipped PPO / Clipped cascade - clipped cascade not used in the paper (not for fixed KL and adaptive KL models)

                    # Calculate policy loss
                    ratios = [torch.exp(curr - old) for (curr, old) in zip(log_probs_cascade, sb.log_probs_cascade)]
                    surr1 = ratios[0] * sb.advantage
                    surr2 = torch.clamp(ratios[0], 1.0 - self.clipranges[0], 1.0 + self.clipranges[0]) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Calculate value loss
                    value_clipped = sb.values_cascade[0] + torch.clamp(value - sb.values_cascade[0], -self.clipranges[0], self.clipranges[0])
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    # Add Kullback-Leibler divergence loss
                    if self.cascade_depth < 2:
                        kl_losses = kl_losses2 = torch.zeros(1)
                    else:
                        # add kl loss with second network in cascade
                        if self.dynamic_neglogpacs:
                            kl_losses = -log_probs_cascade[0] + log_probs_cascade[1]
                            kl_losses2 = torch.clamp(-log_probs_cascade[0], torch.log(1.0-self.clipranges[0])-sb.log_probs_cascade[0], torch.log(1.0+self.clipranges[0])-sb.log_probs_cascade[0]) + log_probs_cascade[1]
                        else:
                            kl_losses = -log_probs_cascade[0] + sb.log_probs_cascade[1]
                            kl_losses2 = torch.clamp(-log_probs_cascade[0], torch.log(1.0-self.clipranges[0])-sb.log_probs_cascade[0], torch.log(1.0+self.clipranges[0])-sb.log_probs_cascade[0]) + sb.log_probs_cascade[1]
                    kl_loss = self.flow_factor*torch.max(kl_losses, kl_losses2).mean()
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss + kl_loss

                    # Construction of additional loss terms for clipped version of PC model (not used for paper)
                    if self.cascade_depth > 1:
                        if self.dynamic_neglogpacs:
                            for k in range(1, self.cascade_depth):
                                hidden_pg_losses = self.mesh_factor*imp_factors[k-1]*(log_probs_cascade[k]-log_probs_cascade[k-1])
                                hidden_pg_losses2 = self.mesh_factor*imp_factors[k-1]*(-log_probs_cascade[k-1]-torch.clamp(
                                    -log_probs_cascade[k], 
                                    torch.log(1.0-self.clipranges[k])-sb.log_probs_cascade[k], 
                                    torch.log(1.0+self.clipranges[k])-sb.log_probs_cascade[k]))
                                if k < (self.cascade_depth-1):
                                    hidden_pg_losses2 += imp_factors[k]*(torch.clamp(
                                        -log_probs_cascade[k], 
                                        torch.log(1.0-self.clipranges[k])-sb.log_probs_cascade[k], 
                                        torch.log(1.0+self.clipranges[k])-sb.log_probs_cascade[k])
                                        +log_probs_cascade[k+1])
                                hidden_pg_loss = torch.max(hidden_pg_losses, hidden_pg_losses2).mean()
                                loss += hidden_pg_loss
                        else:
                            for k in range(1, self.cascade_depth):
                                # We optimize each hidden policy to be close to the policy 
                                # of the adjacent hidden policies at the previous time step.
                                # It is twice as important to be close to 'shallower' policy,
                                # reflecting wider 'tube width' between shallower policies
                                hidden_pg_losses = self.mesh_factor*imp_factors[k-1]*(log_probs_cascade[k] - sb.log_probs_cascade[k-1])
                                hidden_pg_losses2 = self.mesh_factor*imp_factors[k-1]*(-sb.log_probs_cascade[k-1] - torch.clamp(
                                    -log_probs_cascade[k], 
                                    torch.log(1.0-self.clipranges[k])-sb.log_probs_cascade[k], 
                                    torch.log(1.0+self.clipranges[k])-sb.log_probs_cascade[k]))
                                hidden_pg_loss = torch.max(hidden_pg_losses, hidden_pg_losses2).mean()
                                if k < (self.cascade_depth-1):
                                    # Here we assume that importance factors are fixed during one epoch
                                    hidden_pg_losses = imp_factors[k]*(sb.log_probs_cascade[k+1] - log_probs_cascade[k])
                                    hidden_pg_losses2 = imp_factors[k]*(torch.clamp(
                                        -log_probs_cascade[k], 
                                        torch.log(1.0-self.clipranges[k])-sb.log_probs_cascade[k], 
                                        torch.log(1.0+self.clipranges[k])-sb.log_probs_cascade[k])
                                        +sb.log_probs_cascade[k+1])
                                    hidden_pg_loss += torch.max(hidden_pg_losses, hidden_pg_losses2).mean()
                                loss += hidden_pg_loss
                                
                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodels[0].parameters()) ** 0.5
                if self.max_grad_norm is not None:
                    for k in range(self.cascade_depth):
                        torch.nn.utils.clip_grad_norm_(self.acmodels[k].parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update cascade actor-critic (not separate actor/critic optimizers)
                # for k in range(self.cascade_depth):
                #     self.optimizers[k].zero_grad()
                #     batch_loss.backward(retain_graph=True)
                #     if k == 0:
                #         grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodels[k].parameters()) ** 0.5
                #     torch.nn.utils.clip_grad_norm_(self.acmodels[k].parameters(), self.max_grad_norm)

                # # call all optimizers to update
                # for k in range(self.cascade_depth):
                #     self.optimizers[k].step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        # Optimizer scheduler
        if self.scheduler_flag == True:
            self.scheduler.step()

        return logs

    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
