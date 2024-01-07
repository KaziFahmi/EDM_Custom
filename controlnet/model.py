import torch
import pytorch_lightning
from torch import nn
from midi.models.transformer_model import GraphTransformer
import copy
from midi import utils
from controlnet.zero_conv import ZeroConv

class ControlNet(nn.Module):
    def __init__(self,model: GraphTransformer,input_dims:utils.PlaceHolder):
        super().__init__()
        self.model = model
        # copy the model weights and parameters to a new model
        self.trainable_model = copy.deepcopy(model)
        for param in self.trainable_model.parameters():
            param.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = False
        self.input_zero_conv = ZeroConv(input_dims)
        self.output_zero_conv = ZeroConv(input_dims)
    
    def forward(self, Z_t: utils.PlaceHolder, condtion: utils.PlaceHolder):
        non_trainable_output = self.model(Z_t)
        zero_conved_condition = self.input_zero_conv(condtion)
        # Residual connection between zero convolutioned condition and Z_t
        X = Z_t.X + zero_conved_condition.X # Calculating X
        E = Z_t.E + zero_conved_condition.E # Calculating E
        y = Z_t.y + zero_conved_condition.y # Calculating y
        Z_t_norm = torch.norm(Z_t.pos, dim=1, keepdim=True) # Calculating pos
        Z_t_unit = Z_t.pos / Z_t_norm
        condition_norm = torch.norm(zero_conved_condition.pos, dim=1, keepdim=True)
        pos = Z_t_unit * (condition_norm + Z_t_norm) / 2
        node_mask = Z_t.node_mask
        trainable_copy_input = utils.PlaceHolder(X=X, E=E, y=y, pos=pos, charges=None, node_mask=node_mask).mask()
        trainable_copy_output = self.trainable_model(trainable_copy_input)
        zero_conved_output = self.output_zero_conv(trainable_copy_output)
        # Residual connection between zero convolutioned output and non trainable output
        X = non_trainable_output.X + zero_conved_output.X # Calculating X
        E = non_trainable_output.E + zero_conved_output.E # Calculating E
        y = non_trainable_output.y + zero_conved_output.y # Calculating y
        non_trainable_output_norm = torch.norm(non_trainable_output.pos, dim=1, keepdim=True) # Calculating pos
        non_trainable_output_unit = non_trainable_output.pos / non_trainable_output_norm
        zero_conved_output_norm = torch.norm(zero_conved_output.pos, dim=1, keepdim=True)
        pos = non_trainable_output_unit * (zero_conved_output_norm + non_trainable_output_norm) / 2
        node_mask = non_trainable_output.node_mask
        output = utils.PlaceHolder(X=X, E=E, y=y, pos=pos, charges=None, node_mask=node_mask).mask()
        
        return output

