import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from RIM import RIMCell

#NUM_UNITS = 4
#k = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        try:
            m.weight.data.normal_(0, 1)
            m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
            if m.bias is not None:
                m.bias.data.fill_(0)
        except:
            pass

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=True, use_text=False, use_rim=False, num_units=4, k=2, input_heads=1):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.use_rim = use_rim
        if use_rim:
            self.num_units = num_units
            self.k = k
            self.input_heads = input_heads

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        

        # Define memory
        if self.use_memory:
            if use_rim:
                self.memory_rnn = RIMCell(device, self.image_embedding_size, self.semi_memory_size // self.num_units, self.num_units, self.k, 'LSTM', input_value_size = 64, num_input_heads = self.input_heads, comm_value_size = self.semi_memory_size // self.num_units)
                #self.memory_rnn = RIMCell(device, self.image_embedding_size, self.semi_memory_size // self.num_units, self.num_units, self.k, 'LSTM', input_value_size = 64, num_input_heads = 4, comm_value_size = self.semi_memory_size // self.num_units)
            else:
                self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):

        return self.num_units*(self.image_embedding_size // self.num_units) if self.use_rim else self.image_embedding_size
        #return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            if self.use_rim:    
                hidden = list(hidden)
                hidden[0] = hidden[0].view(hidden[0].size(0), self.num_units, -1) 
                hidden[1] = hidden[0].view(hidden[1].size(0), self.num_units, -1)
                x = x.unsqueeze(1)

                hidden = self.memory_rnn(x, hidden[0], hidden[1])
                hidden = list(hidden)
                hidden[0] = hidden[0].view(hidden[0].size(0), -1)
                hidden[1] = hidden[1].view(hidden[1].size(0), -1)
            else:
                hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
