import copy
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import os

from modules.layer.self_atten import SelfAttention
from envs.starcraft.smac_maps import get_map_params
from torch.nn.parameter import Parameter

class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(torch.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return torch.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return torch.squeeze(x, dim=1)

class StrategyQuantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, args):
        super(StrategyQuantizer, self).__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
        # 使用均匀分布初始化，选择合适的min和max值以确保初始化的embeddings尽量分散
        if args.coach_embedding_init == 'uniform':
            self.embeddings.weight.data.uniform_(-1.0 / self.n_embeddings, 1.0 / self.n_embeddings)

        elif args.coach_embedding_init == 'normal':
            self.embeddings.weight.data.normal_(0, 1)
            
        elif args.coach_embedding_init == 'orth':
            nn.init.orthogonal_(self.embeddings.weight)

    def diversity_loss(self):
        # Calculate the diversity loss by computing the pairwise distances between embeddings
        # The pairwise distances are calculated using the 
        # The diversity loss is the sum of the pairwise distances
        embeddings = self.embeddings.weight
        distances = torch.cdist(embeddings, embeddings).mean()
        return 1/distances

    def forward(self, x):
        distances = torch.cdist(x, self.embeddings.weight)
        indices = torch.argmin(distances, dim=-1)
        return self.embeddings(indices)
    

class EmbeddingCoach(nn.Module):
    def __init__(self, args):
        super(EmbeddingCoach, self).__init__()
        self.args = args
        self.state_dim = int(np.prod(args.state_shape)) 
        self.bs = args.batch_size_run
        
        dh = args.coach_hidden_dim
        self.ds = args.n_strategy
        self.na = args.n_agents
        
        
        self.att = SelfAttention(self.state_dim, args.att_heads, args.att_embed_dim)

        self.hypernet_hidden_dim =  args.hypernet_hidden_dim
        self.gat_head = args.gat_heads

        # Graph Attention Network layers
        # Transform layers for allies and enemies to have uniform dimensionality
        self.ally_feats_dim, self.enemy_feats_dim, _, self.n_enemies = self.get_state_shape(args)
        self.ally_transform = nn.Sequential (
            nn.Linear(self.ally_feats_dim, self.hypernet_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hypernet_hidden_dim, self.ally_feats_dim * self.hypernet_hidden_dim * self.gat_head)
        )
        self.enemy_transform = nn.Sequential (
            nn.Linear(self.enemy_feats_dim, self.hypernet_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hypernet_hidden_dim, self.enemy_feats_dim * self.hypernet_hidden_dim * self.gat_head)
        )

        # Embedding and Quantization
        self.de = args.embedding_dim
        self.fc2 = nn.Sequential(
            nn.Linear(self.hypernet_hidden_dim, dh),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dh, self.de)
        )
        self.strategy_h = None
        self.test = None
        # policy for continouos team strategy
        self.dh = dh
        
        self.num_embeddings = self.ds  # Number of discrete embeddings
        self.quantizer = StrategyQuantizer(self.num_embeddings, self.de, args)


        self.decoder = nn.Sequential(
            nn.Linear(self.de, dh),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dh, 2 * dh)
        )
        
        self.prev_embeddings = None
        
        # Attention layers
        self.attetion_weight_mapper = nn.Linear(2 * self.hypernet_hidden_dim, 1)
        self.unify_output_heads_rescue = Merger(self.gat_head, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.activation2 = nn.LeakyReLU(negative_slope=0.2)

    def _build_inputs(self,batch,t):
        inputs = batch["state"][:, t]
        return inputs
    
    def init_hidden(self,batch_size):
        self.hidden_state = self.fc[2].weight.new(1, self.dh).zero_()
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.unsqueeze(0).expand(batch_size, self.na, -1).view(-1,self.dh)  # bav
            
    def build_input_graph(self, state_dicts):
        ally_state = torch.stack([state_dict['ally_state'].clone().detach() for state_dict in state_dicts], dim=0)
        enemy_state = torch.stack([state_dict['enemy_state'].clone().detach() for state_dict in state_dicts], dim=0)
        
        # print ("ally_state", ally_state.shape) # [batch, n_agents, ally_feats_dim]
        # print ("enemy_state", enemy_state.shape) # [batch, n_enemies, enemy_feats_dim]

        ally_hyper_weights = self.ally_transform(ally_state).view(self.bs, self.na, self.ally_feats_dim, self.hypernet_hidden_dim * self.gat_head)
        enemy_hyper_weights = self.enemy_transform(enemy_state).view(self.bs, self.n_enemies, self.enemy_feats_dim, self.hypernet_hidden_dim*self.gat_head)

        # print ("ally_hyper", ally_hyper_weights.shape) # [batch, n_agents, ally_feats_dim, hypernet_hidden_dim * gat_head]
        # print ("enemy_hyper", enemy_hyper_weights.shape) # [batch, n_enemies, enemy_feats_dim, hypernet_hidden_dim * gat_head]

        # [bs, n_agents, ally_feats_dim] * [bs, n_agents, ally_feats_dim, gat_head * hypernet_hidden_dim] -> [bs, n_agents, gat_head * hypernet_hidden_dim]
        ally_embedding = torch.einsum('bna,bnah->bnh', ally_state, ally_hyper_weights).view(self.bs, self.na, self.gat_head, self.hypernet_hidden_dim)
        enemy_embedding = torch.einsum('bne,bneh->bnh', enemy_state, enemy_hyper_weights).view (self.bs, self.n_enemies, self.gat_head, self.hypernet_hidden_dim)

        # print ("ally_embedding", ally_embedding.shape) # [batch, n_agents, gat_head, hypernet_hidden_dim]
        # print ("enemy_embedding", enemy_embedding.shape) # [batch, n_enemies, gat_head, hypernet_hidden_dim]
        
        entity_embedding = torch.cat([ally_embedding, enemy_embedding], dim=1) # [batch, n_agents + n_enemies, gat_head, hypernet_hidden_dim]
        # print ("entity_embedding", entity_embedding.shape)
        
        entity_embedding_repeat = entity_embedding.repeat(1, self.na + self.n_enemies, 1, 1)
        entity_embedding_interleave_repeat = entity_embedding.repeat_interleave(self.na + self.n_enemies, dim=1)
        
        # print ("entity_embedding_repeat", entity_embedding_repeat.shape) # [batch, (n_agents + n_enemies) * (n_agents + n_enemies), gat_head, hypernet_hidden_dim]
        # print ("entity_embedding_interleave_repeat", entity_embedding_interleave_repeat.shape) # [batch, (n_agents + n_enemies) * (n_agents + n_enemies), gat_head, hypernet_hidden_dim]
        
        entities_embedding_concat = self.activation(torch.cat([entity_embedding_repeat, entity_embedding_interleave_repeat], dim=-1))
        # print ("entities_embedding_concat", entities_embedding_concat.shape) # [batch, (n_agents + n_enemies) * (n_agents + n_enemies), gat_head, 2 * hypernet_hidden_dim]
        
        attention_weights = self.attetion_weight_mapper(entities_embedding_concat).view(self.bs, self.na + self.n_enemies, self.na + self.n_enemies, self.gat_head)
        # print ("attention_weights", attention_weights.shape) # [batch, (n_agents + n_enemies), (n_agents + n_enemies), gat_head]
        
        hyper_attention_output = torch.einsum('bijh,bjhf->bihf', attention_weights, entity_embedding) 

        # print ("hyper_attention_output", hyper_attention_output.shape) # [batch, (n_agents + n_enemies), gat_head, hypernet_hidden_dim]
        
        # self.bs , (self.na + self.n_enemies), self.gat_head, self.hypernet_hidden_dim -> self.bs *(self.na + self.n_enemies), self.gat_head, self.hypernet_hidden_dim
        hyper_attention_output = hyper_attention_output.reshape(self.bs * (self.na + self.n_enemies), self.gat_head, self.hypernet_hidden_dim)
        # print ("hyper_attention_output", hyper_attention_output.shape) # [batch * (n_agents + n_enemies), gat_head, hypernet_hidden_dim]
        
        hyper_attention_output = self.unify_output_heads_rescue(hyper_attention_output) # ([bs*(na+ne), hypernet_hidden_dim])
        # print ("hyper_attention_output", hyper_attention_output.shape) # [batch * (n_agents + n_enemies), hypernet_hidden_dim]
        
        hyper_attention_output = hyper_attention_output.view(self.bs, self.na + self.n_enemies, self.hypernet_hidden_dim)
        
        # print ("hyper_attention_output 1", hyper_attention_output.shape) # [batch, (n_agents + n_enemies), hypernet_hidden_dim] 
        # for i in range(self.bs):
        #     for j in range(self.na + self.n_enemies):
        #         print (hyper_attention_output[i][j][:5])
            
            
        hyper_attention_output = self.fc2(hyper_attention_output[:, :self.na, :])

        # print ("hyper_attention_output 2", hyper_attention_output.shape) # [batch, n_agents, n_starategy]
        
        # hyper_attention_output = self.activation2(hyper_attention_output)
        

        # hyper_attention_output = F.softmax(hyper_attention_output, dim=-1) 
        
        # print ("hyper_attention_output", hyper_attention_output.shape) # [batch, n_agents, n_starategy]
        
        hyper_attention_output = hyper_attention_output.view(self.bs * self.na, -1)
        
        return hyper_attention_output
        

    def generate_weights(self, batch, t, args):
        state_dicts = self.get_state_dict_(batch["state"][:, t], args)
        weights = self.build_input_graph(state_dicts) # [batch, n_agents, state_dim]
        return weights

    def strategy(self, mix_weights, is_inference):

        self.strategy_h = mix_weights
        out = self.decoder(mix_weights)
        # print (out.shape)
        mu = out[:, :self.args.coach_hidden_dim]
        log_var = out[:, self.args.coach_hidden_dim:]
        log_var = log_var.clamp_(-10, 0)
        std = (log_var * 0.5).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        # print ("z", z.shape)
        return z, mu, log_var

    def forward(self, batch,t,isinference=False):
        self.bs = batch["state"][:, t].shape[0]
        mix_weights = self.generate_weights(batch,t,self.args) # [batch, n_agents, dh]
        z_team, mu, logvar = self.strategy(mix_weights,isinference)
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
                
        self.quantizer.embeddings.weight.data = self.prev_embeddings


    def get_state_dict_(self, states, args):
        # print ("states", states.shape)
        # for i in range(states.shape[0]):
        #     print (states[i][:5])
        """
        Processes a batch of concatenated state arrays into dictionaries with meaningful keys.

        Args:
        - states (torch.Tensor): A batch of concatenated state arrays.
        - args (Namespace): Configuration containing parameters such as the number of agents,
        number of enemies, shield bits for ally and enemy, unit type bits, etc.

        Returns:
        - list of dicts: A list where each item is a state dictionary for an instance in the batch.
        """
        map_params = get_map_params(args.env_args["map_name"])
        n_agents = map_params["n_agents"]
        n_enemies = map_params["n_enemies"]
        shield_bits_ally = 1 if map_params["a_race"] == "P" else 0
        shield_bits_enemy = 1 if map_params["b_race"] == "P" else 0
        unit_type_bits = map_params["unit_type_bits"]

        nf_al = 4 + shield_bits_ally + unit_type_bits
        nf_en = 3 + shield_bits_enemy + unit_type_bits

        # Calculate the indices for splitting the state vector
        idx_ally = n_agents * nf_al
        idx_enemy = idx_ally + n_enemies * nf_en

        # print ("states", states.shape)
        # for i in range(states.shape[0]):
        #     print (states[i][:5])
        
        batch_size = states.shape[0]
        state_dicts = []

        for i in range(batch_size):
            state = states[i]
            ally_state = state[:idx_ally].reshape(n_agents, nf_al)
            enemy_state = state[idx_ally:idx_enemy].reshape(n_enemies, nf_en)
            remainder = state[idx_enemy:]

            state_dict = {
                'ally_state': ally_state,
                'enemy_state': enemy_state,
            }

            if args.env_args["state_last_action"]:
                last_action_length = n_agents * args.n_actions
                # print ("last_action_length", last_action_length)
                last_action = remainder[:last_action_length].reshape(n_agents, -1)
                state_dict['last_action'] = last_action
                remainder = remainder[last_action_length:]

            if args.env_args["state_timestep_number"]:
                timestep_number = remainder
                state_dict['timestep_number'] = timestep_number
                
            # print ("state_dict last action", state_dict['last_action'].shape)

            state_dicts.append(state_dict)
        

        return state_dicts

    def get_state_shape(self, args):
        """
        Returns the shape of the state vector given the environment arguments.

        Args:
        - args (Namespace): Configuration containing parameters such as the number of agents,
        number of enemies, shield bits for ally and enemy, unit type bits, etc.

        Returns:
        - tuple: The shape of the state vector.
        """
        map_params = get_map_params(args.env_args["map_name"])
        n_agents = map_params["n_agents"]
        n_enemies = map_params["n_enemies"]
        print ("n_agents", n_agents, "n_enemies", n_enemies)
        shield_bits_ally = 1 if map_params["a_race"] == "P" else 0
        shield_bits_enemy = 1 if map_params["b_race"] == "P" else 0
        unit_type_bits = map_params["unit_type_bits"]
        nf_al = 4 + shield_bits_ally + unit_type_bits
        nf_en = 3 + shield_bits_enemy + unit_type_bits
        
        return [nf_al, nf_en, n_agents, n_enemies]

    # visuiallization
    def save_embeddings(self, directory, filename):
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
