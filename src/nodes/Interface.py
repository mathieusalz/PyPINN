import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
import numpy as np
from modulus.sym.graph import Graph
from modulus.sym.key import Key
from modulus.sym.domain.constraint import Constraint
from modulus.sym.distributed import DistributedManager


class Interface(nn.Module):
    
    def __init__(self, inter_nodes, exter_nodes):
        super().__init__()
        self.interior_model = Graph(
            inter_nodes, Key.convert_list(["x", "y", "turning_angle"]), Key.convert_list(["u", "v","rho", "T"])
        )
        self.exterior_model = Graph(
            exter_nodes, Key.convert_list(["x", "y","turning_angle"]), Key.convert_list(["u", "v","rho", "T"])
        )
        self.manager = DistributedManager()
        self.device = self.manager.device
            
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = in_vars["x"]
        y = in_vars["y"]
        angle = in_vars["turning_angle"]
        
        invar = Constraint._set_device(
            in_vars, device=self.device, requires_grad= True
        )
        
        out_1 = self.interior_model(invar)
        out_2 = self.exterior_model(invar)
        
        interface_u = out_1["u"] - out_2["u"]
        interface_v = out_1["v"] - out_2["v"]
        interface_rho = out_1["rho"] - out_2["rho"]
        interface_T = out_1["T"] - out_2["T"]
        
        return {"interface_u": interface_u,
                "interface_v": interface_v,
                "interface_rho": interface_rho, 
                "interface_T": interface_T}
            