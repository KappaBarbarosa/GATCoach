import copy
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.self_atten import SelfAttention

class Coach(nn.Module):
    def __init__(self, args):
        super(Coach, self).__init__()
        self.args = args
        self.state_dim = int(np.prod(args.state_shape)) if (self.args.coach_input == "state") else int(np.prod(args.obs_shape))*args.n_agents
        dh = args.coach_hidden_dim
        self.ds = args.n_strategy
        self.na = args.n_agents
        self.fc1 = nn.Sequential(nn.Linear(self.state_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.na * args.rnn_hidden_dim))
        self.att = SelfAttention(args.rnn_hidden_dim, args.att_heads, args.att_embed_dim)
        self.fc2 = nn.Linear(args.att_heads *  args.att_embed_dim, dh)

        # policy for continouos team strategy
        self.mean = nn.Linear(dh, dh * self.ds)
        self.logvar = nn.Linear(dh, dh * self.ds)
        self.weights = nn.Linear(dh, self.ds)

    def _build_inputs(self,batch,t):
        if self.args.coach_input == "state":
            inputs = batch["state"][:, t].unsqueeze(1)
        else:
            obs = batch["obs"]
            b,T,na,obs_dim = obs.shape 
            inputs = obs.reshape(b,T,self.state_dim)[:, t].unsqueeze(1).float()
        return inputs

    def encode(self, batch,t):
        inputs = self._build_inputs(batch,t) 
        x = self.fc1(inputs).view(-1,self.na, self.args.rnn_hidden_dim)
        att = self.att(x)
        att = F.relu(self.fc2(att), inplace=True).view(-1, self.args.coach_hidden_dim)
        return att

    def strategy(self, h, is_inference):
        # bs, n_agents = h.shape[:2]
        # shared_h = self.shared_layer(h)
        # mu, logvar = self.mean(shared_h), self.logvar(shared_h)
        # mu = mu.view(bs, n_agents, self.ds, -1)
        # logvar = logvar.view(bs, n_agents, self.ds, -1)
        # logvar = logvar.clamp_(-10, 0)

        # mix_weights = F.softmax(self.weights(shared_h), dim=-1)  # [bs, n_agents, n_components]
        
        # # if is_inference:
        # #     max_idx = torch.argmax(mix_weights,dim=-1)
        # #     one_hot = F.one_hot(max_idx, num_classes=self.ds).unsqueeze(-1) 
        # #     mu = (mu * one_hot).sum(dim=-2)
        # #     logvar = (logvar * one_hot).sum(dim=-2)
            
        # # else:
        # mu = torch.sum(mu * mix_weights.unsqueeze(-1), dim=-2)
        # logvar = torch.sum(logvar * mix_weights.unsqueeze(-1), dim=-2)

        # std = (logvar * 0.5).exp()
        # eps = torch.randn_like(std)
        # z = mu + eps * std
        mu, logvar = self.mean(h), self.logvar(h)
        logvar = logvar.clamp_(-10, 0)
        std = (logvar * 0.5).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, batch,t,isinference=False):
        h = self.encode(batch=batch,t=t) # [batch, n_agents, dh]
        z_team, mu, logvar = self.strategy(h,isinference)
        return z_team, mu, logvar
    
    def LLMInput(self,stratigies):
        pass




###############################################################################
#
# Variational Objectives
#
###############################################################################


class VI(nn.Module):
    # I(z^a ; s^a_t+1:t+T-1 | s_t)
    def __init__(self, args):
        super(VI, self).__init__()
        self.args = args

        self.state_dim = int(np.prod(args.obs_shape)) 
        dh = args.coach_hidden_dim
        self.nac = args.n_actions
        self.nag = args.n_agents

        self.fc1 = nn.Sequential(nn.Linear(self.state_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, args.rnn_hidden_dim))
        self.att = SelfAttention(args.rnn_hidden_dim, args.att_heads, args.att_embed_dim)
        self.fc2 = nn.Linear(args.att_heads *  args.att_embed_dim, dh)

        self.mean = nn.Sequential(
            nn.Linear(dh, dh),
            nn.ReLU(),
            nn.Linear(dh, dh))
        self.logvar = nn.Sequential(
            nn.Linear(dh, dh),
            nn.ReLU(),
            nn.Linear(dh, dh))
        self.dh = dh

    def forward(self, batch, Z):
        T = batch.max_seq_length
        O = batch["obs"]
        log_prob = 0
        for t in range(T-1):
            o = O[:,t]
            z = Z[:,int(t/self.args.coach_update_freq)].reshape(-1,self.nag, self.dh)
            #no, ne
            #prev_a = torch.zeros_like(A[:,0]) if t == 0 else A[:,t-1]
            #prev_a = self.action_embedding(prev_a)
            ha = self.fc1(o) # [batch, n_agents, dh]
            h = self.att(ha) # [batch, n_agents, dh]
            h = F.relu(self.fc2(h), inplace=True)
            mu, logvar = self.mean(h), self.logvar(h)
            logvar = logvar.clamp_(-10, 0)
            q_t = D.normal.Normal(mu, (0.5 * logvar).exp())
            log_prob += q_t.log_prob(z).clamp_(-1000, 0).sum(-1)
            
        return -log_prob.mean()
