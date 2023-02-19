self._transformer_module = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

# these are the equivalent to my self.transformer_blocks
for p in self._transformer_module.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

if tfixup:
    # this is the equivalent to my self.linear_embedding
    for p in self._obs_embedding.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, 0, d_model ** (- 1. / 2.))

    temp_state_dic = {}
    for name, param in self._obs_embedding.named_parameters():
        if 'weight' in name:
            temp_state_dic[name] = ((9* num_encoder_layers) ** (- 1. / 4.)) * param

    for name in self._obs_embedding.state_dict():
        if name not in temp_state_dic:
            temp_state_dic[name] = self._obs_embedding.state_dict()[name]
    self._obs_embedding.load_state_dict(temp_state_dic)    

    temp_state_dic = {}
    # these are the equivalent to my self.transformer_blocks
    for name, param in self._transformer_module.named_parameters():
        if any(s in name for s in ["linear1.weight", "linear2.weight", "self_attn.out_proj.weight"]):
            temp_state_dic[name] = (0.67 * (num_encoder_layers) ** (- 1. / 4.)) * param
        elif "self_attn.in_proj_weight" in name:
            temp_state_dic[name] = (0.67 * (num_encoder_layers) ** (- 1. / 4.)) * (param * (2**0.5))

    for name in self._transformer_module.state_dict():
        if name not in temp_state_dic:
            temp_state_dic[name] = self._transformer_module.state_dict()[name]
    self._transformer_module.load_state_dict(temp_state_dic)

if self._policy_head_input == "latest_memory":
    self._policy_head_input_dim = d_model
elif self._policy_head_input == "mixed_memory": # working memory + episodic memory
    self._policy_head_input_dim = 2*d_model
elif self._policy_head_input == "full_memory":
    self._policy_head_input_dim = d_model * self._obs_horizon