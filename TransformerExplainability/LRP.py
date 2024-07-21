"""
Credit for implementation:
https://github.com/hila-chefer/Transformer-Explainability/

Notes:

1. Modifications
Anything where the algorithm is modified or changed in some way will be denoted with a `MODIFIED` tag.

2. Questions
Any questions that I have will be denoted with a `QUESTION` tag.
"""

import torch
from torch import nn
import numpy as np
from einops import rearrange
from model import SimpleSeqClassifier as ComparisonModel

BANDDEF = np.load('banddef.npy')
DROP_WAVELENGTHS = [float(i) for i in open('drop_wavelengths.txt', 'r').readlines()]

dropbands = []
for wl in DROP_WAVELENGTHS:
    deltas = np.abs(BANDDEF - wl)
    dropbands.append(np.argmin(deltas))
BANDDEF_DROPPED_BANDS = torch.tensor(np.delete(BANDDEF, dropbands))

if torch.cuda.is_available():
    device = torch.device(f"cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps") # Apple silicon
else:
    device = torch.device("cpu")


class BandEncoding(nn.Module):
    """Concatenate band wavelength to reflectance spectra."""

    def __init__(self):
        super().__init__()
        self.banddef = torch.unsqueeze(BANDDEF_DROPPED_BANDS, -1).to(dtype=torch.float32)
        #self.banddef = BANDDEF
        self.banddef = ((self.banddef - torch.mean(self.banddef)) / torch.std(self.banddef)).to(device)

    def forward(self, spectra:torch.tensor):
        """ 
            spectra: (b, s, 1)
            banddef: (s, 1)
        """
        if len(spectra.size()) != 3:
            spectra = spectra.unsqueeze(-1)
        encoded = torch.cat((spectra, self.banddef.unsqueeze(0).expand_as(spectra)), dim=-1)
        return encoded
    
###################### Misc. Funcs. ######################

def safe_divide(a:torch.tensor, b:torch.tensor):
    """
    Exact same as a division, but with a small epsilon added to the denominator to avoid division by zero.
    """
    a = a.to(device)
    b = b.to(device)
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = (den + den.eq(0).to(torch.float32) * 1e-9).to(device)
    return a / den * b.ne(0).type(torch.float32)

#############################################################

######################## Define Hooks #######################
## The purpose is to be able to capture both the forward and backward input/output

def forward_hook(self: torch.nn.Module, inp:tuple, output:torch.tensor):
    """
    :param self: the model or sub module that the hook is attached to
    :param inp: input tensor to the forward pass
        Shape: (batch size, ...)
    :param output: output tensor from the forward pass after being processed by the forward method
        Shape: (batch size, ...)
    """
    if type(inp[0]) in (list, tuple):
        self.forward_input = []
        for i in inp[0]:
            x = i.detach()
            x.requires_grad = True
            self.forward_input.append(x)
    else:
        self.forward_input = inp[0].detach()
        self.forward_input.requires_grad = True
        
    self.forward_output = output


def backward_hook(self: torch.nn.Module, grad_input:torch.tensor, grad_output:torch.tensor):
    self.grad_input = grad_input
    self.grad_output = grad_output   

#############################################################                 

####################### RelProp Modules ######################
class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()

        # `forward_hook` will be called when the forward pass is executed
        self.register_forward_hook(forward_hook) 

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha):
        return R
    
class RelPropSimple(RelProp):
    def relprop(self, R, alpha):
        """
        R: relevance score. It's a batch_size x n_dimension tensor that contains the relevance score for each dimension of the output.
        """
        forward_output = self.forward(self.forward_input)
        S = safe_divide(R, forward_output)                        
        C = self.gradprop(forward_output, self.forward_input, S)

        if self._get_name() == "einsum":
            # The einsum module is a special case where the forward_input is a tuple
            outputs = [self.forward_input[0] * C[0], self.forward_input[1] * C[1]]

        elif self._get_name() != "einsum" and torch.is_tensor(self.forward_input) == False:
            print(self._get_name())
            raise Exception(f"Unhandled type: {type(self.forward_input)}")
        
        else:
            # NOTE: This is the normal case (any non-einsum module)
            outputs = self.forward_input * (C[0])
        
        return outputs
    
class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)
    
    def relprop(self, R, alpha):
        # NOTE: this is the residual work that page 785 (actual page 4) in the paper adresses
        # Issue is that the residual connection can cause the relevancy scores to explode so we need to scale them down
        forward_output = self.forward(self.forward_input)
        S = safe_divide(R, forward_output)
        C = self.gradprop(forward_output, self.forward_input, S)

        a = self.forward_input[0] * C[0]
        b = self.forward_input[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs
    
class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha):
        Z = []
        for _ in range(self.num):
            Z.append(self.forward_input)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.forward_input, S)[0]

        R = self.forward_input * C

        return R
    
class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha):
        """
        R: relevance score.
            On the first call, it is the one-hot vector for the target class.
            It's the same dimansion as the OUTPUT of the layer.

        alpha: parameter that controls the amount of positive relevance that is propagated. 
            For only positive relevance propagation, alpha should be 1.
            For only negative relevance propagation, alpha should be 0.

        Returns:
            R: relevance score for the input of the layer.
            It's the same dimension as the INPUT of the layer.
        """
        beta = alpha - 1
        positive_weights = torch.clamp(self.weight, min=0) # weights that positively contribute to the class
        negative_weights = torch.clamp(self.weight, max=0) # weights that negatively contribute to the class
        positive_fw_input = torch.clamp(self.forward_input, min=0)
        negative_fw_input = torch.clamp(self.forward_input, max=0)

        # NOTE: `f` is equivalent to equation 2 in the paper, but broken up into positive/negative parts
        def f(w1, w2, x1, x2) -> torch.tensor:
            # QUESTION: WHY ISN'T THE BIAS INCLUDED IN THIS?
            x1_x_w1 = nn.functional.linear(x1, w1) # essentially x1@w1.T --> the forward feature map (no bias) for pos/neg layer inputs with pos/neg parts of the weights
            # pos_weigth@pos_input = how much does this layer agree with the positive features from the input
            # neg_weigth@pos_input = how much does this layer DISagree with the positive features from the input
            # WILL BE THE SHAPE OF THE OUTPUT OF THE LAYER

            x2_x_w2 = nn.functional.linear(x2, w2) # essentially x2@w2.T
            # neg_weigth@neg_input = how much does this layer agree with the negative features from the input
            # pos_weigth@neg_input = how much does this layer DISagree with the negative features from the input
            # WILL BE THE SHAPE OF THE OUTPUT OF THE LAYER

            # QUESTION: Is this implementation correct? Why are we dividing by (x1_x_w1 + x2_x_w2) instead of x1_x_w1 or x2_x_w2?
            # QUESTION: Why do we need to divide here?
            # MODIFIED:
            S1 = safe_divide(R, x1_x_w1)# x2_x_w2)      # R/(x1@w1.T + x2@w2.T) --> what does this specifically mean? Relevancy scaled by the pos/neg values? Closer to zero+magnitude = ?? because it isn't class specific input
            S2 = safe_divide(R, x2_x_w2)# x1_x_w1)      # R/(x1@w1.T + x2@w2.T)

            #                  output, input, tangent (vector to multiply by the Jacobian matrix of the function)
            C1 = torch.autograd.grad(x1_x_w1, x1, S1)[0]
            C2 = torch.autograd.grad(x2_x_w2, x2, S2)[0]
            # This is the jacobian of the weights (x1@w1.T or x2@w2.T) with respect to x1 or x2 inputs multiplied by the normalized relevancy score
            # This is essentially mapping the relevancy score back on to the input space
            # dy/dx * S

            C1 = x1 * C1
            C2 = x2 * C2

            return C1 + C2

        # pos_weigth@pos_input, neg_weigth@neg_input <-- how much does this layer agree with the positive/negative features from the input
        # you can view this as: how much does the layer agree with the positive/negative parts of the input
        agreement_relevances = f(positive_weights, negative_weights, positive_fw_input, negative_fw_input)
        # ned_weigth@pos_input, pos_weigth@neg_input <-- given the input is positive, how much is the layer supressing that positive input
        # you can view this as: how much does the layer disagree with the positive/negative parts of the input
        disagreement_relevances = f(negative_weights, positive_weights, positive_fw_input, negative_fw_input)

        # NOTE:
        # Linear(input) is equivalent to the following:
        #       positive_fw_input = torch.clamp(input, min=0) -- parts of input > 0
        #       negative_fw_input = torch.clamp(input, max=0) -- parts of input < 0
        #       positive_weights  = torch.clamp(Linear.weight, max=0) -- parts of Linear.weight > 0
        #       negative_weights  = torch.clamp(Linear.weight, max=0) -- parts of Linear.weight < 0
        #
        # pp = positive_fw_input @ positive_weights.T
        # nn = negative_fw_input @ negative_weights.T
        # pn = positive_fw_input @ negative_weights.T
        # np = negative_fw_input @ positive_weights.T
        #
        # (np + nn + pn + pp) + Linear.bias == Linear(input)

        R = alpha * agreement_relevances - beta * disagreement_relevances

        # assert torch.isclose(R.sum(), torch.tensor(1.0)), f"Relevance score needs to sum to 1. Got {R.sum().item()} instead."

        # print(R.sum().item())

        return R
    
class LayerNorm(nn.LayerNorm, RelProp):
    # Note: `relprop` is not implemented for LayerNorm in the original implementation
    pass

# Won't be used since we aren't training this model
# class Dropout(nn.Dropout, RelProp):
#     pass

class GELU(nn.GELU, RelProp):
    pass

class Tanh(nn.Tanh, RelProp): 
    # Think we could so the same thing here as with the GELU as they both allow for negative values
    pass

class Softmax(nn.Softmax, RelProp):
    pass

class einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        # self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x

    def relprop(self, cam, alpha):
        # cam = self.drop.relprop(cam, alpha)
        cam = self.fc2.relprop(cam, alpha)
        cam = self.act.relprop(cam, alpha)
        cam = self.fc1.relprop(cam, alpha)
        return cam


class Conv1d(nn.Conv1d, RelProp):
    
    def relprop(self, R, alpha):
        beta = alpha - 1
        positive_weights = torch.clamp(self.weight, min=0) # weights that positively contribute to the class
        negative_weights = torch.clamp(self.weight, max=0) # weights that negatively contribute to the class
        positive_fw_input = torch.clamp(self.forward_input, min=0)
        negative_fw_input = torch.clamp(self.forward_input, max=0)

        def f(w1, w2, x1, x2):
            # MODIFIED: Conv1d was not implemented in the original implementation. Originally, it was `conv2d`
            Z1 = torch.nn.functional.conv1d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
            Z2 = torch.nn.functional.conv1d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
            Z1 = torch.transpose(Z1, 1, 2)
            Z2 = torch.transpose(Z2, 1, 2)
            S1 = safe_divide(R, Z1)
            S2 = safe_divide(R, Z2)
            C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            C2 = x2 * self.gradprop(Z2, x2, S2)[0]
            return C1 + C2

        agreement_relevances = f(positive_weights, negative_weights, positive_fw_input, negative_fw_input)
        disagreement_relevances = f(negative_weights, positive_weights, positive_fw_input, negative_fw_input)

        R = alpha * agreement_relevances - beta * disagreement_relevances
        return R

class Attention(nn.Module):
    """
    We have to implement the attention from scratch due to PyTorch's implementation of the attention mechanism.
    This implementation is stright from https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/ViT_LRP.py
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        # self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        # attn = self.attn_drop(attn)

        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        # out = self.proj_drop(out)
        return out

    def relprop(self, cam, alpha):
        # cam = self.proj_drop.relprop(cam, alpha)
        cam = self.proj.relprop(cam, alpha)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, alpha)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        # cam1 = self.attn_drop.relprop(cam1, alpha)
        cam1 = self.softmax.relprop(cam1, alpha)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, alpha)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, alpha)
    

class Block(nn.Module):

    def __init__(self, dim, num_heads, dim_feedforward, qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim_feedforward, drop=drop)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        # TODO: NEED TO ADD IN AN OPTION FOR NORM FIRST OR AFTER
        x1, x2 = self.clone1(x, 2)
        x = self.norm1(self.add1([x1, self.attn(x2)]))
        x1, x2 = self.clone2(x, 2)
        x = self.norm2(self.add2([x1, self.mlp(x2)]))
        return x

    def relprop(self, cam, alpha):
        (cam1, cam2) = self.add2.relprop(cam, alpha)
        cam2 = self.mlp.relprop(cam2, alpha)
        cam2 = self.norm2.relprop(cam2, alpha)
        cam = self.clone2.relprop((cam1, cam2), alpha)

        (cam1, cam2) = self.add1.relprop(cam, alpha)
        cam2 = self.attn.relprop(cam2, alpha)
        cam2 = self.norm1.relprop(cam2, alpha)
        cam = self.clone1.relprop((cam1, cam2), alpha)
        return cam

#############################################################

class LrpSimpleSeqClassifier(nn.Module):
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

        self.project = Conv1d(
            in_channels=2,
            out_channels=dim_proj,
            kernel_size=ker_proj, 
            stride=str_proj,
            padding=0,
            dilation=1)

        self.gelu = GELU()
        self.tanh = Tanh()

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=dim_proj,
        #     nhead=n_heads,
        #     dim_feedforward=dim_ff,
        #     batch_first=True,
        #     activation='gelu',
        #     dropout=dropout,
        #     norm_first=False
        # )

        # self.transformer_encoder = nn.TransformerEncoder(
        #     encoder_layer,
        #     num_layers=n_layers,
        # )
        encoder_layer = Block(dim_proj, 
                              n_heads, 
                              dim_ff, 
                              drop=dropout
                              )
        self.transformer_encoder = nn.Sequential(*[encoder_layer for _ in range(n_layers)])

        self.aggregate = agg

        self.classifier = Linear(dim_proj, n_classes)

        key_map = {
            'transformer_encoder.layers.0.self_attn.in_proj_weight': 'transformer_encoder.0.attn.qkv.weight',
            'transformer_encoder.layers.0.self_attn.in_proj_bias': 'transformer_encoder.0.attn.qkv.bias',
            'transformer_encoder.layers.0.self_attn.out_proj.weight': 'transformer_encoder.0.attn.proj.weight',
            'transformer_encoder.layers.0.self_attn.out_proj.bias': 'transformer_encoder.0.attn.proj.bias',
            'transformer_encoder.layers.0.linear1.weight': 'transformer_encoder.0.mlp.fc1.weight',
            'transformer_encoder.layers.0.linear1.bias': 'transformer_encoder.0.mlp.fc1.bias',
            'transformer_encoder.layers.0.linear2.weight': 'transformer_encoder.0.mlp.fc2.weight',
            'transformer_encoder.layers.0.linear2.bias': 'transformer_encoder.0.mlp.fc2.bias',
            'transformer_encoder.layers.0.norm1.weight': 'transformer_encoder.0.norm1.weight',
            'transformer_encoder.layers.0.norm1.bias': 'transformer_encoder.0.norm1.bias',
            'transformer_encoder.layers.0.norm2.weight': 'transformer_encoder.0.norm2.weight',
            'transformer_encoder.layers.0.norm2.bias': 'transformer_encoder.0.norm2.bias'
        }
        
        weights = torch.load(load_model_weights, map_location=device)
        new_state_dict = {key_map.get(k, k): v for k, v in weights.items()}
        self.load_state_dict(new_state_dict)
        self.eval()
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.transpose(x, 1, 2)
        x = x.to(device=device, dtype=torch.float)
        x = self.project(x)
        x = self.tanh(x)
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

    def relprop(self, cam, alpha):
        cam = self.classifier.relprop(cam, alpha)
        # cam = self.aggregate.relprop(cam, alpha)
        reverse_layers = reversed(self.transformer_encoder) if isinstance(self.transformer_encoder, nn.Sequential) else reversed(self.transformer_encoder.layers)
        for layer in reverse_layers:
            cam = layer.relprop(cam, alpha)
        cam = self.tanh.relprop(cam, alpha)
        cam = self.project.relprop(cam, alpha) # this is the Conv1D
        return cam

#############################################################

def generate_LRP(model:nn.Module, input:torch.tensor, index:int=None):
        """
        :param model: the model to be explained
        :param input: the input data to the model that you want explained
        :param index: the index of the class you want explained (default is the class with the highest probability)
        """
        output = model(input).to(device) # get logits for each class
        alpha = 1 # initialize alpha to 1, this is the parameter that controls the amount of relevance that is propagated
        if index is None:
            index = int(torch.argmax(output))

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = torch.tensor(one_hot) # one hot vector for the target class
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
        one_hot = torch.sum(one_hot * output) # gets the output logits for the target class (i.e. for 3 class [1.345, 0, 0])

        model.zero_grad()
        one_hot.backward(retain_graph=True) # this will use the PyTorch autograd to get the model gradients for ONLY the target class (since all other classes are zeroed out)

        return model.relprop(one_hot_vector, alpha)

if __name__ == "__main__":
    random_input = torch.tensor([[-3.3285e-01, -4.8020e-01,  1.1949e+00, -2.7834e-02,  1.2051e-02,
          5.1867e-01,  7.8318e-01, -5.2424e-02, -1.5914e+00,  7.4032e-01,
         -5.2807e-01, -2.0163e-01, -7.6042e-01,  1.4143e+00, -1.3339e-01,
         -1.1488e+00, -1.6353e-01,  2.8325e-02, -4.0749e-01, -2.4021e+00,
          1.1789e+00, -2.0609e+00,  1.8353e-01, -9.2517e-01, -5.5572e-01,
         -3.9764e-02, -8.8691e-01,  5.0612e-01,  1.2979e+00, -5.3953e-01,
         -3.1716e-01,  1.1224e-01,  4.0154e-01, -1.6492e+00, -6.2814e-02,
          1.3582e-01, -7.0032e-01,  1.8469e+00, -1.6156e+00, -1.8856e+00,
         -1.2411e+00, -1.2918e+00,  1.8438e+00, -1.9191e+00, -6.1699e-01,
         -1.0245e+00,  1.6796e-03, -1.0375e-01,  4.7901e-02, -1.4730e+00,
          1.0681e+00,  1.2437e+00, -9.2954e-02, -3.3647e-01,  8.2542e-01,
         -1.3794e+00,  1.0435e+00,  1.6399e+00,  1.1914e-01,  5.6414e-01,
          4.7066e-01,  6.4112e-01,  1.5320e+00,  4.5399e-01,  6.2346e-01,
          1.3017e+00,  4.7891e-01,  1.1071e+00,  9.3005e-01,  1.0855e+00,
          1.4212e-01, -2.1617e+00,  9.1375e-01,  7.2524e-01,  9.8355e-01,
         -1.1550e-01, -1.3525e+00,  5.3810e-01,  9.8600e-01, -7.0099e-01,
          1.0574e+00, -6.6555e-01,  6.0678e-02, -2.1952e-01,  1.6445e+00,
          4.6636e-01, -2.0898e-03,  6.4502e-01,  2.3511e-01, -7.1286e-01,
          1.1351e+00,  4.7033e-01,  1.3971e+00, -2.0729e-01, -1.9885e+00,
          3.1910e-02, -2.3332e-01, -1.0171e+00, -5.1097e-01, -1.0522e+00,
         -4.3109e-01,  1.3563e-01,  7.6662e-01,  1.1692e+00, -7.8312e-02,
          3.4622e-01, -2.1428e+00,  1.1630e-01,  1.5320e+00,  1.2081e+00,
         -1.3538e+00,  1.4952e+00,  1.7818e+00,  8.3709e-01, -5.9895e-02,
         -8.3600e-01, -2.6688e-01, -1.5594e+00, -5.1278e-01,  1.3655e+00,
          1.2563e+00,  1.8008e+00, -1.1659e+00,  9.7394e-01, -7.8751e-01,
         -3.1114e-01, -8.4213e-01,  6.2460e-02,  6.6546e-01, -1.8649e-01,
         -9.0970e-01,  1.1386e+00,  1.1437e+00, -8.4357e-01,  1.5948e+00,
         -1.5947e+00,  1.7762e-01,  1.0170e-01,  1.6731e+00, -9.2353e-03,
         -1.0291e+00, -7.9390e-01,  2.6272e-01, -4.6026e-01,  5.0015e-01,
         -7.6769e-02, -1.8031e+00,  2.1947e-01,  2.5361e-01, -2.7244e-01,
         -3.8312e-01,  7.4371e-01,  7.9157e-01, -4.4206e-01, -4.5435e-01,
         -3.7227e-01,  1.0443e+00,  8.0706e-01, -2.6660e-01,  1.4897e-01,
          3.1159e-01, -9.4035e-01,  1.5273e+00,  8.0519e-01,  1.9341e+00,
         -6.1135e-02,  9.0619e-01,  1.0645e+00,  6.7172e-01, -4.3349e-01,
          1.4240e+00,  1.0987e-01,  8.2980e-01,  1.7810e+00,  6.4075e-02,
          1.3202e+00, -6.4645e-01, -5.7470e-01, -8.5054e-01, -2.1109e-01,
         -1.7627e-01,  4.0658e-01,  9.1878e-01,  2.2106e-01,  3.9400e-01,
          7.5865e-01,  1.2968e+00,  3.3584e-01, -3.6463e-01,  3.3895e-01,
         -1.5676e+00,  1.2007e+00, -1.5040e+00, -1.3092e+00,  1.2737e+00,
         -1.9130e-01,  1.3202e+00,  4.7609e-01, -3.1145e+00, -5.6814e-01,
         -8.7998e-01, -3.4557e-01,  1.1099e+00, -7.6349e-01,  3.8518e-01,
          3.4646e-02,  1.4998e+00, -1.1634e+00,  6.9450e-01,  8.0679e-01,
         -1.2908e+00,  9.0041e-01, -5.9882e-01,  4.3863e-01,  6.9382e-01,
          3.3251e-01, -1.9927e+00, -8.5779e-01, -9.3867e-01, -5.6776e-01,
         -4.4706e-01,  1.3245e+00,  1.0621e+00,  1.5102e+00, -4.3076e-01,
         -1.3847e+00,  9.2650e-01, -4.5302e-01, -9.1205e-01, -2.0595e-01,
         -9.1200e-01,  2.0716e+00, -9.6457e-01, -1.9851e+00,  1.1219e+00,
          1.5657e+00, -8.4759e-01,  8.5975e-01,  3.4537e-01,  1.2683e+00,
         -9.4482e-02, -1.2093e+00,  8.5826e-01,  8.3149e-01,  6.0692e-01,
         -3.3517e-01,  3.2598e-01, -2.5383e+00,  6.3167e-01,  2.1285e+00,
         -1.3085e+00,  2.2202e+00,  9.6717e-01,  2.9839e-01,  6.7004e-01,
         -1.9295e+00, -6.6299e-01,  6.4013e-01, -9.6179e-01, -5.4375e-01,
         -1.1630e+00,  3.2914e-01, -1.6098e+00, -1.1161e+00,  6.3476e-01,
         -1.4198e-01, -2.0061e+00, -6.0511e-01, -3.7937e-01, -1.0107e+00,
          1.2552e+00,  1.7819e+00,  5.2402e-01,  1.8801e-01,  9.8489e-01,
          2.0129e+00,  1.7401e+00,  1.2891e+00]]).to(device)
    
    model = LrpSimpleSeqClassifier(
        ker_proj = 9,
        str_proj = 5,
        dim_proj = 512,
        n_layers = 1,
        n_heads  = 8,
        dim_ff   = 128,
        n_classes= 3,
        dropout  = 0.1,
        agg      = 'max',
        load_model_weights = 'jake_v3data_weights.pt'
    )
    model.to(device)
    model.eval()
    logits = model(random_input)

    comparison_model = ComparisonModel(
        ker_proj = 9,
        str_proj = 5,
        dim_proj = 512,
        n_layers = 1,
        n_heads  = 8,
        dim_ff   = 128,
        n_classes= 3,
        dropout  = 0.1,
        agg      = 'max',
        load_model_weights = 'jake_v3data_weights.pt'
    )
    comparison_model.to(device)
    comparison_model.eval()
    comparison_logits = comparison_model(random_input)

    print(logits.tolist())
    print(comparison_logits.tolist())

    assert torch.allclose(logits, comparison_logits, atol=1e-5), "Model outputs are not the same!"

    cam = generate_LRP(model, random_input)
    cam = cam.sum(1) # the output will be of shape (batch_size, 2, N bands), so sum along the 1st dimension to get the relevance score for each band
    print(cam)
