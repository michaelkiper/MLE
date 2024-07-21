import torch
from torch import nn
from torch import tensor 

class RMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms

class ResidualBlock(nn.Module):
    ## Question: should I replace all of the batch norms with layer norms???
    def __init__(self, in_dim, out_dim, activation=nn.GELU(), dropout_rate=0.1, norm_type='batch'):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        # self.bn1 = nn.BatchNorm1d(out_dim)  # BatchNorm after first linear layer
        self.bn1 = self._get_normalization_layer(norm_type, out_dim)
        
        self.linear2 = nn.Linear(out_dim, out_dim)
        # self.bn2 = nn.BatchNorm1d(out_dim)  # BatchNorm after second linear layer
        self.bn2 = self._get_normalization_layer(norm_type, out_dim)
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        
        # If the input dim and output dim don't alaign, need to project it into output dimension space 
        # If it's the same, then it's just the identity matrix
        self.project = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        if in_dim != out_dim:
            self.bn_project = self._get_normalization_layer(norm_type, out_dim)

    def _get_normalization_layer(self, norm_type, num_features):
        norm_type = norm_type.lower()
        if norm_type == 'batch':
            return nn.BatchNorm1d(num_features)
        elif norm_type == 'layer':
            return nn.LayerNorm(num_features)
        elif norm_type == 'rms':
            return RMSNorm(num_features)
        else:
            raise ValueError("Invalid normalization type. Choose 'batch', 'layer', or 'rms'.")
    
    def forward(self, x):
        identity = self.project(x)
        if hasattr(self, 'bn_project'):
            identity = self.bn_project(identity)
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out += identity
        out = self.activation(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, in_dim:int, layer_dims:list, num_classes:int, dropout_rate=0.1, norm_type='batch'):
        super(ResNet, self).__init__()
        
        layer_dims = [in_dim]+layer_dims
        self.residual_blocks = nn.Sequential()
        for i in range(len(layer_dims)-1):
            self.residual_blocks.add_module(
                f"block_{i}", 
                ResidualBlock(layer_dims[i], layer_dims[i+1], dropout_rate=dropout_rate, norm_type=norm_type)
                )

        self.mlp_classification_head = nn.Linear(layer_dims[-1], num_classes)
    
    def forward(self, x:tensor):
#         x = x.unsqueeze(1) # add in a dummy dim for the spectral dims
        x = self.residual_blocks(x)
        x = self.mlp_classification_head(x)
        return x

def make_model(arch, input_dim, num_classes):
    layers = []
    for key in arch['layers'].keys():
        layers += [arch['layers'][key]['dim'] for _ in range(arch['layers'][key]['count'])]
    dropout = arch['dropout']
    norm_type = arch['norm_type']

    model = ResNet(input_dim, layers, num_classes, dropout, norm_type)
    return model
