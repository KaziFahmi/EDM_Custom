import torch
import torch.nn as nn

class ZeroConv(nn.module):
    def __init__(self, input_dims: utils.PlaceHolder):
        super().__init__()
        
        self.input_X_zero_conv = nn.Linear(input_dims['X'], input_dims.X + input_dims.charges)
        with torch.no_grad():
            nn.init.constant_(self.input_X_zero_conv.weight, 0)
            nn.init.constant_(self.input_X_zero_conv.bias, 0)
            
        self.input_E_zero_conv = nn.Linear(input_dims['E'], input_dims.E)
        with torch.no_grad():
            nn.init.constant_(self.input_E_zero_conv.weight, 0)
            nn.init.constant_(self.input_E_zero_conv.bias, 0)
            
        self.input_y_zero_conv =  nn.Linear(input_dims['y'], input_dims.y)
        with torch.no_grad():
            nn.init.constant_(self.input_y_zero_conv.weight, 0)
            nn.init.constant_(self.input_y_zero_conv.bias, 0)
            
        self.input_pos_zero_conv = nn.Linear(input_dims['pos'])
        with torch.no_grad():
            nn.init.constant_(self.input_pos_zero_conv.weight, 0)
            nn.init.constant_(self.input_pos_zero_conv.bias, 0)
        
    
    def forward(self, features: utils.PlaceHolder):
        X = self.input_X_zero_conv(features.X)
        E = self.input_E_zero_conv(features.E)
        y = self.input_y_zero_conv(features.y)
        pos = self.input_pos_zero_conv(features.pos)
        out = utils.PlaceHolder(X=X, E=E, y=y, pos=pos, charges=None, node_mask=features.node_mask)
        return out
        