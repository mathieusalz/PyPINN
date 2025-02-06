from .Geometry import Geometry
from .Network import Network
from typing import Union, List, Dict
from modulus.sym.domain.constraint import PointwiseConstraint
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.domain import Domain
import numpy as np
from modulus.sym.node import Node

from .nodes.Interface import Interface
from .nodes.PlotterNode import PlotterNode
from .plotters.CVP_comp_Val_nonScaled import CVP_Comp_Val 

class Pairing:
    """
    Represents the pairing of geometries and networks.

    :param geometry: The geometry or geometries to be paired.
    :type geometry: Union[Geometry, tuple]
    :param inputs: List of input keys for the pairing.
    :type inputs: list
    :raises RuntimeError: If the geometry is not assigned a network or interface.
    """

    def __init__(self, 
                 geometry: Union[Geometry, tuple], 
                 inputs: list):
        self.geometry = geometry
        
        if not isinstance(geometry, list):
            if not hasattr(geometry, 'network') and not hasattr(geometry, 'interface'):
                raise RuntimeError(f"{geometry.name} was not assigned a network nor is an interface")
            else:
                if hasattr(geometry, 'network'):
                    self.network = geometry.network
                else:
                    self.network = geometry.interface
        else:
            for geo in geometry:
                if not hasattr(geo, 'network') and not hasattr(geo, 'interface'):
                    raise RuntimeError(f"{geo.name} was not assigned a network nor is an interface")
        
        self.inputs = inputs
                                                
class Constraint(Pairing):
    """
    Represents constraints in the computational domain.

    :param geometry: The geometry or geometries to be constrained.
    :type geometry: Union[Geometry, tuple]
    :param constraint_type: Type of the constraint ('PDE', 'CFD', 'BOUND', 'VELOCITY', 'INTERFACE').
    :type constraint_type: str
    :param batch_size: Batch size for the constraint, defaults to 100.
    :type batch_size: int, optional
    :param global_weight: Global weight for the constraint, defaults to 1.
    :type global_weight: Union[float, int], optional
    :param output_weights: Output weights for the constraint, defaults to None.
    :type output_weights: dict, optional
    :param sdf_weighting: Whether to use SDF weighting, defaults to None.
    :type sdf_weighting: Union[bool, None], optional
    :param inputs: List of input keys for the constraint, defaults to None.
    :type inputs: list, optional
    :param parametric: Parametric settings for the constraint, defaults to None.
    :type parametric: dict, optional
    :param name: Name of the constraint, defaults to None.
    :type name: str, optional
    :raises RuntimeWarning: If the batch size is too large.
    """

    def __init__(self, 
                 geometry: Union[Geometry, tuple], 
                 constraint_type: str,
                 batch_size: int = 100,
                 global_weight: Union[float, int] = 1,
                 output_weights=None,
                 sdf_weighting: Union[bool, None] = None,
                 inputs: List[str] = None,
                 parametric : dict=None,
                 name : str = None):
        super().__init__(geometry=geometry, inputs=inputs)
        
        self.constraint_type = constraint_type
                        
        if "CFD" in self.constraint_type:
            self.geometry = geometry.parameter_select(parametric)
            
        if constraint_type == 'INTERFACE':
            self.name = self.network[0].name + "_" + "Interface" + "_" + self.network[1].name
        else:
            self.name = self.geometry.name + "_" + self.network.name + "_" + self.constraint_type

        if geometry.N <= batch_size:
            RuntimeWarning(f"{geometry.name} batch size too large, set to value of {geometry.N//2}")
            self.batch_size = geometry.N // 2
        else: 
            self.batch_size = batch_size
            
        self.global_weight = global_weight
        self.output_weights = output_weights
        self.sdf_weighting = sdf_weighting
        
    def __str__(self):
        return f'Name: {self.name} \t Batch Size: {self.batch_size} \t Weight: {self.global_weight} \t SDF: {self.sdf_weighting}' 
                 
    def create_modObj(self, mod_domain : Domain):
        """
        Creates a Modulus constraint object and adds it to the domain.

        :param mod_domain: The Modulus domain to which the constraint will be added.
        :type mod_domain: object
        """
        if self.constraint_type == 'PDE':
            outputs = ['continuity', 'momentum_x', 'momentum_y', 'energy']
            nodes = self.network.nodes

        elif self.constraint_type == 'CFD' or self.constraint_type == 'BOUND':
            outputs = ['u', 'v', 'rho', 'T']
            nodes = self.network.nodes

        elif self.constraint_type == 'VELOCITY':
            outputs = ['u', 'v']
            nodes = self.network.nodes

        elif self.constraint_type == 'INTERFACE':
            outputs = ["interface_u", "interface_v", "interface_rho", "interface_T"]
            inter_nodes = self.network[0].nodes
            outer_nodes = self.network[1].nodes

            interface_node = Node(inputs=self.inputs, 
                                  outputs=outputs, 
                                  evaluate=Interface(inter_nodes=inter_nodes, exter_nodes=outer_nodes), 
                                  name="Interface")

            nodes = [interface_node]

        outvar = {}
        lambda_weighting = {}

        for output in outputs:
            outvar[output] = getattr(self.geometry, output)

            if self.output_weights is not None and output in self.output_weights:
                weight = self.global_weight * self.output_weights[output]
            else:
                weight = self.global_weight

            if self.sdf_weighting:
                weight = self.geometry.sdf * weight
            else:
                weight = np.ones((self.geometry.N, 1)) * weight

            lambda_weighting[output] = weight

        invar = {}

        for inp in self.inputs:
            invar[inp] = getattr(self.geometry, inp)

        mod_constraint = PointwiseConstraint.from_numpy(nodes=nodes,
                                                        invar=invar,
                                                        outvar=outvar,
                                                        lambda_weighting=lambda_weighting,
                                                        batch_size=self.batch_size)

        mod_domain.add_constraint(mod_constraint, self.name)
        
class Plot(Pairing):
    """
    Represents plots in the computational domain.

    :param geometry: The geometry or geometries to be plotted.
    :type geometry: Union[Geometry, tuple]
    :param inputs: List of input keys for the plot, defaults to None.
    :type inputs: list, optional
    :param x_min: Minimum x-coordinate for the plot, defaults to None.
    :type x_min: float, optional
    :param x_max: Maximum x-coordinate for the plot, defaults to None.
    :type x_max: float, optional
    :param y_min: Minimum y-coordinate for the plot, defaults to None.
    :type y_min: float, optional
    :param y_max: Maximum y-coordinate for the plot, defaults to None.
    :type y_max: float, optional
    :param name: Name of the plot, defaults to None.
    :type name: str, optional
    :param validator: Validator object for the plot, defaults to None.
    :type validator: object, optional
    """

    def __init__(self, 
                 geometry: Union[Geometry, tuple], 
                 inputs : List[str] = None,
                 x_min : float = None,
                 x_max : float = None,
                 y_min : float = None,
                 y_max : float = None,
                 name : str=None,
                 validator : Geometry=None):
        super().__init__(geometry=geometry, inputs=inputs)
        
        self.name = name + "__" + "Plot"
        self.validator = validator
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
    def create_modObj(self, mod_domain : Domain):
        """
        Creates a Modulus plot object and adds it to the domain.

        :param mod_domain: The Modulus domain to which the plot will be added.
        :type mod_domain: object
        """
        outputs = ["u", "v", "rho", "T"]
        outputs.extend(self.inputs)
                    
        plotter_node = [Node(inputs=self.inputs, 
                             outputs=outputs, 
                             evaluate=PlotterNode(plot=self))]

        invar_numpy = {inp: np.random.rand(10) for inp in self.inputs}

        inferencer = PointwiseInferencer(nodes=plotter_node,
                                         invar=invar_numpy,
                                         output_names=outputs,
                                         plotter=CVP_Comp_Val(self))

        mod_domain.add_inferencer(inferencer, self.name)
