# Contains model implementations.

import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    # return number of parameters
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_grad(self):
        # More efficient than optimizer.zero_grad() according to:
        # Szymon Migacz "PYTORCH PERFORMANCE TUNING GUIDE" at GTC-21.
        # - doesn't execute memset for every parameter
        # - memory is zeroed-out by the allocator in a more efficient way
        # - backward pass updates gradients with "=" operator (write) (unlike
        # zero_grad() which would result in "+=").
        # In PyT >= 1.7, one can do `model.zero_grad(set_to_none=True)`
        for p in self.parameters():
            p.grad = None

    def print_params(self):
        for p in self.named_parameters():
            print(p)


class MLPModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size , hidden_size)
        self.fc2 = nn.Linear(hidden_size , num_classes)

    def forward(self, x):
        # Assume linealized input: here ‘images.view(−1, 28*28)‘.
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class NeuralNetModel(BaseModel):
    def __init__(self, input_feat_dim, emb_dim, hidden_size, num_classes,
                 num_layers=1, dropout_rate=0.0, use_projection=False,
                 use_resnet=False, use_layernorm=False):
        super().__init__()
        # note that if use_projection is True, we'll get 1 + `num_layers`

        self.num_layers = num_layers

        if num_layers == 0:  # logistic regression baseline
            self.output_layer = nn.Linear(input_feat_dim, num_classes)
        else:
            if use_projection:
                self.input_layers = nn.Sequential(
                    nn.Linear(input_feat_dim, emb_dim),  # input layer
                    nn.Dropout(dropout_rate),
                    nn.ReLU(),
                    nn.Linear(emb_dim, hidden_size), # projection layer
                    nn.Dropout(dropout_rate),
                    nn.ReLU(inplace=True)
                )
            else:
                self.input_layers = nn.Sequential(
                    nn.Linear(input_feat_dim, hidden_size),  # input layer
                    nn.Dropout(dropout_rate),
                    nn.ReLU(inplace=True)
                )

            # add more layers if num_layers > 1
            # self.hidden_layers is identity if num_layers == 1
            hidden_layer_list = []
            for _ in range(num_layers-1):
                if use_resnet:
                    hidden_layer_list.append(
                        ResidualFFlayers(hidden_size, dropout_rate, use_layernorm=use_layernorm))
                else:
                    hidden_layer_list.append(nn.Linear(hidden_size, hidden_size))
                    hidden_layer_list.append(nn.Dropout(dropout_rate))
                    # relu was removed
                    hidden_layer_list.append(nn.ReLU(inplace=True))
            self.hidden_layers = nn.Sequential(*hidden_layer_list)

            self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.num_layers > 0:
            x = self.input_layers(x)
            x = self.hidden_layers(x)
            x = self.output_layer(x)
            return x
        else:  # logistic regression
            return self.output_layer(x)


# A layer of residual nets
# layer norm = False seems to be much better for this dataset
class ResidualFFlayers(nn.Module):
    def __init__(self, hidden_dim, dropout, use_layernorm=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        self.basic_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    # pre-layer norm resnet
    def forward(self, x):
        out = self.layer_norm(x) if self.use_layernorm else x
        return self.basic_layer(out) + x


# A block of residual feed-forward layers in Transformer
class TransformerFFlayers(nn.Module):
    def __init__(self, inner_dim, res_dim, dropout, use_layernorm=True,
                 use_res=True):
        super().__init__()

        self.res_dim = res_dim
        self.inner_dimim = inner_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.use_res = use_res

        self.ff_layers = nn.Sequential(
            nn.Linear(res_dim, inner_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, res_dim),
            nn.Dropout(dropout),
        )

        if use_layernorm:
            self.layer_norm = nn.LayerNorm(res_dim)

    def forward(self, x):
        out = self.layer_norm(x) if self.use_layernorm else x
        if self.use_res:
            out = self.ff_layers(out) + x
        else:
             out = self.ff_layers(out)
        return out
