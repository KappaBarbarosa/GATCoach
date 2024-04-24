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
        self.state_dim = int(np.prod(args.state_shape)) 
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
        inputs = batch["state"][:, t]
        return inputs

    def encode(self, batch,t):
        inputs = batch["state"][:, t].unsqueeze(1)
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


# class VI(nn.Module):
#     # I(z^a ; s^a_t+1:t+T-1 | s_t)
#     def __init__(self, args):
#         super(VI, self).__init__()
#         self.args = args

#         self.state_dim = int(np.prod(args.state_shape)) 
#         dh = args.coach_hidden_dim
#         self.na = args.n_actions
        
#         self.action_embedding = nn.Embedding(self.na, self.na)
#         self.action_embedding.weight.data = torch.eye(self.na).to(args.device)
#         for p in self.action_embedding.parameters():
#             p.requires_grad = False

#         self.fc1 = nn.Linear(self.state_dim + self.na, args.rnn_hidden_dim)
#         self.att = SelfAttention(self.rnn_hidden_dim, args.att_heads, args.att_embed_dim)
#         self.fc2 = nn.Linear(args.att_heads *  args.att_embed_dim, args.dh)

#         self.mean = nn.Sequential(
#             nn.Linear(dh, dh),
#             nn.ReLU(),
#             nn.Linear(dh, dh))
#         self.logvar = nn.Sequential(
#             nn.Linear(dh, dh),
#             nn.ReLU(),
#             nn.Linear(dh, dh))
#         self.dh = dh

#     def forward(self, batch, z_t0):
#         T = batch.max_seq_length
#         O = batch["state"]
#         A = batch["actions"]
#         H = []
#         z0 = None
#         log_prob = 0
#         for t in range(T-1):
#             o = O[:,t]
#             #no, ne
#             #prev_a = torch.zeros_like(A[:,0]) if t == 0 else A[:,t-1]
#             #prev_a = self.action_embedding(prev_a)
#             a = self.action_embedding(A[:,t])
#             o = torch.cat([o, a], -1)
#             ha = self.fc1(o) # [batch, n_agents, dh]

#             h = self.mha(x, m) # [batch, n_agents, dh]
#             mu, logvar = self.mean(h), self.logvar(h)
#             logvar = logvar.clamp_(-10, 0)
#             q_t = D.normal.Normal(mu[ma], (0.5 * logvar[ma]).exp())
#             log_prob += q_t.log_prob(z_t0).clamp_(-1000, 0).sum(-1)
#         return -log_prob.mean()
