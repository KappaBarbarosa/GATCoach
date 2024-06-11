import copy
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.self_atten import SelfAttention

class StrategyQuantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, args):
        super(StrategyQuantizer, self).__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
        # 使用均匀分布初始化，选择合适的min和max值以确保初始化的embeddings尽量分散
        if args.coach_embedding_init == 'uniform':
            self.embeddings.weight.data.uniform_(-1, 1)

        elif args.coach_embedding_init == 'normal':
            self.embeddings.weight.data.normal_(0, 1)
            
        elif args.coach_embedding_init == 'orth':
            nn.init.orthogonal_(self.embeddings.weight)

    def diversity_loss(self):
        # Calculate the diversity loss by computing the pairwise distances between embeddings
        # The pairwise distances are calculated using the L2 norm
        # The diversity loss is the sum of the pairwise distances
        embeddings = self.embeddings.weight
        distances = torch.cdist(embeddings, embeddings).mean()
        return 1/distances

    def forward(self, x):
        distances = torch.cdist(x, self.embeddings.weight)
        indices = torch.argmin(distances, dim=-1)
        return self.embeddings(indices)
    

class HPNCoach(nn.Module):
    def __init__(self, args):
        super(HPNCoach, self).__init__()
        self.args = args
        self.state_dim = int(np.prod(args.state_shape)) 
        
        dh = args.coach_hidden_dim
        self.ds = args.n_strategy
        self.na = args.n_agents
        self.att = SelfAttention(self.state_dim, args.att_heads, args.att_embed_dim)

        self.fc = nn.Sequential(nn.Linear(args.att_heads *  args.att_embed_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.na * dh))

        # Embedding and Quantization
        self.de = args.embedding_dim
        self.fc2 = nn.Linear(dh, self.de)
        
        # policy for continouos team strategy
        self.dh = dh
        
        self.num_embeddings = self.ds  # Number of discrete embeddings
        self.quantizer = StrategyQuantizer(self.num_embeddings, self.de, args)
        self.decoder = nn.Sequential(
            nn.Linear(self.de, dh),
            nn.ReLU(),
            nn.Linear(dh, 2 * dh)
        )
        self.prev_embeddings = None

    def _build_inputs(self,batch,t):
        inputs = batch["state"][:, t]
        return inputs
    
    def init_hidden(self,batch_size):
        self.hidden_state = self.fc[2].weight.new(1, self.dh).zero_()
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.unsqueeze(0).expand(batch_size, self.na, -1).view(-1,self.dh)  # bav
            
    def encode(self, batch,t):
        inputs = batch["state"][:, t].unsqueeze(1) ## [batch, 1, state_dim]
        print("batch : ", batch["state"][:, t].shape)
        att = self.att(inputs) ## [batch, 1, att_heads * att_embed_dim]
        x = self.fc(att).view(-1, self.dh) ## [batch, n_agents, dh]
        h = F.relu(self.fc2(x), inplace=True) ## [batch, n_agents, dh]
        return h
    
    def compute_distances(self, x):
        distances = torch.cdist(x, self.quantizer.embeddings.weight)
        weights = F.softmax(-8*distances, dim=-1)
        return weights
    
    def strategy(self, h, is_inference):
        mix_weights = self.compute_distances(h)
        strategy_h = torch.matmul(mix_weights, self.quantizer.embeddings.weight)
        out = self.decoder(strategy_h)
        # print (out.shape)
        mu = out[:, :self.args.coach_hidden_dim]
        log_var = out[:, self.args.coach_hidden_dim:]
        log_var = log_var.clamp_(-10, 0)
        std = (log_var * 0.5).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        # print ("z : ", z.shape)
        return z, mu, log_var

    def forward(self, batch,t,isinference=False):
        h = self.encode(batch=batch,t=t) # [batch, n_agents, dh]
        z_team, mu, logvar = self.strategy(h,isinference)
        return z_team, mu, logvar
    
    def loss(self):
        diversity = self.quantizer.diversity_loss()
        return diversity
    
    def update_embeddings(self):

        if self.prev_embeddings is None:
            self.prev_embeddings = self.quantizer.embeddings.weight.detach().clone()
        else:
            decay_rate = 0.99
            with torch.no_grad():
                self.prev_embeddings *= decay_rate
                self.prev_embeddings += (1 - decay_rate) * self.quantizer.embeddings.weight.detach()






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
