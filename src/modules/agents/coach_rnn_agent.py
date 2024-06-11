import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.nn import LayerNorm
from utils.th_utils import orthogonal_init_
class CRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.q_head = nn.Linear(args.rnn_hidden_dim + args.coach_hidden_dim, args.n_actions)
        self.z_team = None

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def set_team_strategy(self, z_team):
        self.z_team = z_team

    def set_part_team_strategy(self, z_team, indices):
        prev = self.z_team
        self.z_team = z_team * indices.unsqueeze(-1) + \
                prev * (1-indices.unsqueeze(-1))

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        # print (inputs.size())
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)
        
        if b == self.z_team.shape[0] * 2: # imaginary
            z_ = self.z_team.repeat(2,1,1)
        else:
            z_ = self.z_team

        if getattr(self.args, "use_layer_norm", False):
            hh = self.layer_norm(hh)
            h_ = th.cat([hh, z_], -1)
            q = self.q_head(h_)
        else:
            h_ = th.cat([hh, z_], -1)
            q = self.q_head(h_)

        return q.view(b, a, -1), hh.view(b, a, -1)