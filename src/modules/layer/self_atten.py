import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class SelfAttention(nn.Module):
    def __init__(self, input_size, heads, embed_size):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.toqueries = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.tovalues = nn.Linear(self.input_size, self.emb_size * heads, bias = False)

    def forward(self, x):
        b, t, hin = x.size()
        assert hin == self.input_size, f'Input size {{hin}} should match {{self.input_size}}'
        
        h = self.heads 
        e = self.emb_size
        
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*h, t, t)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return out
    
class MHA(nn.Module):
    """
    the class of Multi-Head Attention
    """
    def __init__(self, input_dim, hidden_dim, n_heads,n_agents):
        super(MHA, self).__init__()
        self.encode = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(input_dim, hidden_dim))
        self.WQs = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, hidden_dim)) for i in range(n_heads)])
        self.WKs = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, hidden_dim)) for i in range(n_heads)])
        self.WVs = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, hidden_dim)) for i in range(n_heads)])
        self.scale =  1. / np.sqrt(hidden_dim)
        self.n_agents = n_agents

    def forward(self, x):
        """
        x:      [batch, n_entities, input_dim]
        ma:     [batch, n_agents, n_all]
        return: [batch, n_agents, hidden_dim*n_heads]
        """

        h = self.encode(x) # [batch, n, hidden_dim]
        ha = h[:,:self.n_agents].contiguous()

        outputs = []
        for WQ, WK, WV in zip(self.WQs, self.WKs, self.WVs):
            Q = (ha @ WQ.unsqueeze(0)) # [batch, na, hidden_dim]
            K = (h @ WK.unsqueeze(0))  # [batch, n, hidden_dim]
            V = (h @ WV.unsqueeze(0))  # [batch, n, hidden_dim]
            QK_T = Q.bmm(K.transpose(1,2)) * self.scale # [batch, na, n]

            QK_T = F.softmax(QK_T, dim=-1)
            prob = QK_T / (QK_T.sum(-1, keepdims=True) + 1e-12)
            if torch.isnan(QK_T).sum() > 0:
                import pdb; pdb.set_trace()

            z = prob.bmm(V) # [batch, na, hidden_dim]
            outputs.append(z.unsqueeze(1))
        output = torch.cat(outputs, dim=1) # [batch, n_heads, na, hidden_dim]
        return output.mean(1) # [batch, na, hidden_dim]