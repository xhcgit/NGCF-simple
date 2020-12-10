import torch as t
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class MODEL(nn.Module):
    def __init__(self, args, userNum, itemNum, hide_dim, layer=[16,16]):
        super(MODEL, self).__init__()
        self.args = args
        self.userNum = userNum
        self.itemNum = itemNum
        self.hide_dim = hide_dim
        self.layer = [hide_dim] + layer
        self.embedding_dict = self.init_weight(userNum, itemNum, hide_dim)
        # GCN activation is leakyReLU
        slope = self.args.slope
        self.act = t.nn.LeakyReLU(negative_slope=slope)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.layer)-1):
                self.layers.append(GCNLayer(self.layer[i], self.layer[i+1], weight=True, activation=self.act))
    
    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(t.empty(itemNum, hide_dim))),
        })
        return embedding_dict
    
    def forward(self, adj):
        all_user_embeddings = [self.embedding_dict['user_emb']]
        all_item_embeddings = [self.embedding_dict['item_emb']]
        if len(self.layers) == 0:
            return self.embedding_dict['user_emb'], embedding_dict['item_embed']

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(adj, self.embedding_dict['user_emb'], self.embedding_dict['item_emb'])
            else:
                embeddings = layer(adj, embeddings[: self.userNum], embeddings[self.userNum: ])
            
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_user_embeddings += [norm_embeddings[: self.userNum]]
            all_item_embeddings += [norm_embeddings[self.userNum: ]]

        user_embedding = t.cat(all_user_embeddings, 1)
        item_embedding = t.cat(all_item_embeddings, 1)
        return user_embedding, item_embedding


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, weight=True, activation=None):
        super(GCNLayer, self).__init__()
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(t.Tensor(in_dim, out_dim))
            self.v_w = nn.Parameter(t.Tensor(in_dim, out_dim))
            init.xavier_uniform_(self.u_w)
            init.xavier_uniform_(self.v_w)
        self.act = activation

    def forward(self, adj, user_feat, item_feat):
        user_feat = t.mm(user_feat, self.u_w)
        item_feat = t.mm(item_feat, self.v_w)
        feat = t.cat([user_feat, item_feat], dim=0)
        feat = self.act(t.spmm(adj, feat))
        return feat