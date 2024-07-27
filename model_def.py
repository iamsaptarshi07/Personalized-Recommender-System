import torch
from torch import nn
import torch.nn.functional as F

class NeuCF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_user = config['num_users']
        self.num_items = config['num_items']
        self.embedding_dim = config['embedding_dim']
        self.hidden_layers = config['hidden_layers']
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.user_embedding = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        input_dim = self.embedding_dim * 2
        self.fc_layer = nn.ModuleList()

        for hidden_dim in self.hidden_layers:
            self.fc_layer.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.dropout = nn.Dropout(self.dropout_rate)
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=-1)

        for layer in self.fc_layer:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.output_layer(x)
        x = F.sigmoid(x)
        return x
