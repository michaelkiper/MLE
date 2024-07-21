import torch
from torch import nn
import numpy as np

BANDDEF = np.load('banddef.npy')
DROP_WAVELENGTHS = [float(i) for i in open('drop_wavelengths.txt', 'r').readlines()]

dropbands = []
for wl in DROP_WAVELENGTHS:
    deltas = np.abs(BANDDEF - wl)
    dropbands.append(np.argmin(deltas))
BANDDEF_DROPPED_BANDS = torch.tensor(np.delete(BANDDEF, dropbands))

############# Get device #############
if torch.cuda.is_available():
    device = torch.device(f"cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps") # Apple silicon
else:
    device = torch.device("cpu")
######################################

############### Model ################
class BandEncoding(nn.Module):
    """Concatenate band wavelength to reflectance spectra."""

    def __init__(self):
        super().__init__()
        self.banddef = torch.unsqueeze(BANDDEF_DROPPED_BANDS, -1).to(dtype=torch.float32)
        #self.banddef = BANDDEF
        self.banddef = ((self.banddef - torch.mean(self.banddef)) / torch.std(self.banddef)).to(device)

    def forward(self, spectra):
        """ 
            spectra: (b, s, 1)
            banddef: (s, 1)
        """
        if len(spectra.size()) != 3:
            spectra = spectra.unsqueeze(-1)
        encoded = torch.cat((spectra, self.banddef.unsqueeze(0).expand_as(spectra)), dim=-1)
        return encoded

class SimpleSeqClassifier(nn.Module):
    def __init__(self, 
                 ker_proj: int = 9,
                 str_proj: int = 4,
                 dim_proj: int = 128,
                 n_layers: int = 1,
                 n_heads: int = 32,
                 dim_ff: int = 128,
                 n_classes:int = 3,
                 dropout: float = 0.0,
                 agg: str = 'max',
                 load_model_weights:str = None
                 ):
        super().__init__()
        self.ker_proj = ker_proj
        self.str_proj = str_proj

        self.encoder = BandEncoding()

        self.project = nn.Conv1d(
            in_channels=2,
            out_channels=dim_proj,
            kernel_size=ker_proj, 
            stride=str_proj,
            padding=0,
            dilation=1)

        self.gelu = torch.nn.functional.gelu

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_proj,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
            activation='gelu',
            dropout=dropout
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.aggregate = agg

        self.classifier = nn.Linear(dim_proj, n_classes)

        self.initialize_weights()
        
        if load_model_weights is not None:
            weights = torch.load(load_model_weights, map_location=device)
            self.load_state_dict(weights)
            self.eval()
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.transpose(x, 1, 2)
        x = x.to(device=device, dtype=torch.float)
        x = self.project(x)
        x = torch.tanh(x)
        x = torch.transpose(x, 1, 2)
        x = self.transformer_encoder(x)
        if self.aggregate == 'mean':
            x = torch.mean(x, dim=1)
        elif self.aggregate == 'max':
            x,_ = torch.max(x, dim=1)
        elif self.aggregate == 'flat':
            x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
######################################