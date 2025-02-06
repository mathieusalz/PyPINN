from shapely.geometry import Point,Polygon
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
import numpy as np
from modulus.sym.graph import Graph
from modulus.sym.key import Key
from modulus.sym.domain.constraint import Constraint
from modulus.sym.distributed import DistributedManager

class PlotterNode(nn.Module):
    
    def __init__(self, plot):
        super().__init__()
        
        self.geometry = plot.geometry
        self.inputs = plot.inputs

        outputs = ["u","v","rho", "T"]

        if isinstance(self.geometry, list):
            self.N_regions = len(self.geometry)

            for geo in self.geometry:
                if not hasattr(geo.network, 'graph'):
                    net_graph  = Graph(geo.network.nodes, 
                                    Key.convert_list(self.inputs), 
                                    Key.convert_list(outputs)
                                    )
                    geo.network.graph = net_graph
            
        else:
            self.N_regions = 1
            
            if not hasattr(self.geometry.network, 'graph'):
                net_graph  = Graph(self.geometry.network.nodes, 
                                Key.convert_list(self.inputs), 
                                Key.convert_list(outputs)
                                )
                self.geometry.network.graph = net_graph
                
        self.manager = DistributedManager()
        self.device = self.manager.device
            
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        
        out_vars = {"u": None,
                    "v": None,
                    "rho": None,
                    "T": None
                   }
        
        for inp in self.inputs:
            out_vars[inp] = None
        
        for i in range(self.N_regions): 
            
            if self.N_regions == 1:
                geo = self.geometry
                net = geo.network
            else:
                geo = self.geometry[i]
                net = geo.network

            in_vars = {}

            for inp in self.inputs:
                in_vars[inp] = torch.tensor(getattr(geo, inp)).cuda()

            invar = Constraint._set_device(
                in_vars, device=self.device, requires_grad= False
            )

            output = net.graph(invar)
                            
            for key, value in out_vars.items():
                if key in output.keys():
                    # Move tensor to CPU and convert to numpy array
                    output_array = output[key].cpu().numpy()
                    if value is None:
                        out_vars[key] = output_array
                    else:
                        out_vars[key] = np.concatenate((value, output_array))
                else:
                    # Move tensor to CPU and convert to numpy array
                    invar_array = invar[key].cpu().numpy()
                    
                    if value is None:
                        out_vars[key] = invar_array
                    else:
                        out_vars[key] = np.concatenate((value, invar_array))
            
        for key, value in out_vars.items():
            if value is not None:
                out_vars[key] = torch.tensor(value).to(self.device)
                        
        return out_vars