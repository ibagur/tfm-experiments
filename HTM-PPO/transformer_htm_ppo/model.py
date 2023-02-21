import numpy as np
import torch

from torch.distributions import Categorical
from torch import nn
from torch.nn import functional as F

from transformer import Transformer
from transformer_htm import HTMTransformer
from transformer_htm_noln import HTMTransformerNoLN

class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape, max_episode_length):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
            max_episode_length {int} -- The maximum number of steps in an episode
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.memory_layer_size = config["transformer"]["embed_dim"]
        self.observation_space_shape = observation_space.shape
        self.max_episode_length = max_episode_length

        # Observation encoder
        if len(self.observation_space_shape) > 1:
            # Case: visual observation is available
            # Visual encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, 4,)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
            # Compute output size of convolutional layers
            self.conv_out_size = self.get_conv_output(observation_space.shape)
            in_features_next_layer = self.conv_out_size
        else:
            # Case: vector observation is available
            in_features_next_layer = observation_space.shape[0]
        
        # Hidden layer
        self.lin_hidden = nn.Linear(in_features_next_layer, self.memory_layer_size)
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # Transformer Blocks
        if config["transformer"]["htm"]:
            if config["tfixup"]:
                self.transformer = HTMTransformerNoLN(config["transformer"], self.memory_layer_size, self.max_episode_length)
            else:
                self.transformer = HTMTransformer(config["transformer"], self.memory_layer_size, self.max_episode_length)
        else:
            self.transformer = Transformer(config["transformer"], self.memory_layer_size, self.max_episode_length)
        
        #TEST T-fixup (from: https://github.com/luckeciano/transformers-metarl/blob/trmrl-torch/src/garage/torch/policies/gaussian_transformer_encoder_policy.py)
        # use tfixup with Adam!
        if config["tfixup"]:
            # Xavier initialization  for all parameters excluding input embeddings
            for p in self.transformer.transformer_blocks.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
            # Gaussian initialization for input embeddings
            for p in self.transformer.linear_embedding.parameters():
                if p.dim() > 1:
                    torch.nn.init.normal_(p, 0, config["transformer"]["embed_dim"] ** (- 1. / 2.))            
            # Scale input embeddings in encoder
            temp_state_dic = {}
            for name, param in self.transformer.linear_embedding.named_parameters():
                if 'weight' in name:
                    temp_state_dic[name] = ((9* config["transformer"]["num_blocks"]) ** (- 1. / 4.)) * param

            for name in self.transformer.linear_embedding.state_dict():
                if name not in temp_state_dic:
                    temp_state_dic[name] = self.transformer.linear_embedding.state_dict()[name]
            self.transformer.linear_embedding.load_state_dict(temp_state_dic)

            temp_state_dic = {}
            # Scale v and w matrics in encoder attention block and weight matrices in encoder MLP block
            for name, param in self.transformer.transformer_blocks.named_parameters():
                if any(s in name for s in ["fc.0.weight", "fc_input.0.weight", "fc_input.2.weight", "attn.to_out.weight"]):
                    temp_state_dic[name] = (0.67 * config["transformer"]["num_blocks"] ** (- 1. / 4.)) * param
                elif any(s in name for s in ["attn.to_summary_queries.weight", "attn.to_summary_keys.weight", "attn.attn.to_q.weight", "attn.attn.to_kv.weight"]):
                #elif "self_attn.in_proj_weight" in name:
                    temp_state_dic[name] = (0.67 * config["transformer"]["num_blocks"] ** (- 1. / 4.)) * (param * (2**0.5))

            for name in self.transformer.transformer_blocks.state_dict():
                if name not in temp_state_dic:
                    temp_state_dic[name] = self.transformer.transformer_blocks.state_dict()[name]
            self.transformer.transformer_blocks.load_state_dict(temp_state_dic)            

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(self.memory_layer_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(self.memory_layer_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=self.hidden_size, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)
            
        # Value function
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs:torch.tensor, memory:torch.tensor, memory_mask:torch.tensor, memory_indices:torch.tensor):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations
            memory {torch.tensor} -- Episodic memory window
            memory_mask {torch.tensor} -- Mask to prevent the model from attending to the padding
            memory_indices {torch.tensor} -- Indices to select the positional encoding that matches the memory window

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value function: Value
        """
        # Set observation as input to the model
        h = obs
        # Forward observation encoder
        if len(self.observation_space_shape) > 1:
            batch_size = h.size()[0]
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))

        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))
        
        # Forward transformer blocks
        h, memory = self.transformer(h, memory, memory_mask, memory_indices)

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]
        
        return pi, value, memory

    def get_conv_output(self, shape:tuple) -> int:
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def get_grad_norm(self):
        """Returns the norm of the gradients of the model.
        
        Returns:
            {dict} -- Dictionary of gradient norms grouped by layer name
        """
        grads = {}
        if len(self.observation_space_shape) > 1:
            grads["encoder"] = self._calc_grad_norm(self.conv1, self.conv2, self.conv3)  
            
        grads["linear_layer"] = self._calc_grad_norm(self.lin_hidden)
        
        transfomer_blocks = self.transformer.transformer_blocks
        for i, block in enumerate(transfomer_blocks):
            grads["transformer_block_" + str(i)] = self._calc_grad_norm(block)
        
        for i, head in enumerate(self.policy_branches):
            grads["policy_head_" + str(i)] = self._calc_grad_norm(head)
        
        grads["lin_policy"] = self._calc_grad_norm(self.lin_policy)
        grads["value"] = self._calc_grad_norm(self.lin_value, self.value)
        grads["model"] = self._calc_grad_norm(self, self.value)
          
        return grads
    
    def _calc_grad_norm(self, *modules):
        """Computes the norm of the gradients of the given modules.

        Arguments:
            modules {list} -- List of modules to compute the norm of the gradients of.

        Returns:
            {float} -- Norm of the gradients of the given modules. 
        """
        grads = []
        for module in modules:
            for name, parameter in module.named_parameters():
                #TEST transformer architecture
                if parameter.grad is not None:
                    grads.append(parameter.grad.view(-1))            
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None