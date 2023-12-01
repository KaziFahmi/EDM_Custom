import torch
import pytorch_lightning
from torch import nn
from midi.models.transformer_model import GraphTransformer
import copy
from midi import utils
class ControlNetM(nn.Module):
    def __init__(self,model: GraphTransformer,input_dims:utils.PlaceHolder):
        super().__init__()
        self.model = model
        # copy the model weights and parameters to a new model
        self.trainable_model = copy.deepcopy(model)
        for param in self.trainable_model.parameters():
            param.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.register_parameter('input_zero_conv',nn.Parameter(torch.zeros(1,input_dims.pos)))
    
    def forward(self, Z_t: utils.PlaceHolder, condtion: utils.PlaceHolder):
        return None

