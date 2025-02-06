from modulus.sym.models.activation import Activation
from .architectures.ModFullyConnected import ModFullyConnectedArch
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym.models.modified_fourier_net import ModifiedFourierNetArch
from modulus.sym.key import Key
from modulus.sym.domain.constraint import PointwiseConstraint

from .equations.CustomNavierStokes_noDivide_nd_ideal import CustomNavierStokes
from .equations.CustomNavierStokes_noDivide_nd_ideal_inviscid import CustomNavierStokes as CustomNavierStokes_inviscid

from typing import Dict, List, Tuple

class Network:
    """
    A class to represent and manage neural network architectures for fluid dynamics simulations.

    Attributes:
        name (str): Name of the network.
        architecture (str): Type of network architecture ('FeedForward', 'Fourier', etc.).
        nr_layers (int): Number of layers in the network.
        layer_size (int): Size of each layer in the network.
        activation (str): Activation function to use in the network.
        adaptive (bool): Whether to use adaptive activations.
        config (str): Configuration of the network ('single', 'dual', 'total').
        full_freqs (bool): Whether to use full frequencies in Fourier networks.
        periodic (bool): Whether the network is periodic.
        inputs (list): List of input keys for the network.
        nodes (list): List of nodes in the network.
    """

    def __init__(self,
                 name : str = None,
                 architecture : str ='FeedForward',
                 nr_layers : int = 5,
                 layer_size : int =100,
                 activation : str ='tanh',
                 adaptive : bool =False,
                 config : str ='single',
                 full_freqs : bool =False, 
                 periodic : bool =False,
                 inputs : List[str] =None):
        """
        Initializes a new instance of the Network class.

        Args:
            name (str, optional): Name of the network.
            architecture (str, optional): Type of network architecture.
            nr_layers (int, optional): Number of layers in the network.
            layer_size (int, optional): Size of each layer in the network.
            activation (str, optional): Activation function to use in the network.
            adaptive (bool, optional): Whether to use adaptive activations.
            config (str, optional): Configuration of the network.
            full_freqs (bool, optional): Whether to use full frequencies in Fourier networks.
            periodic (bool, optional): Whether the network is periodic.
            inputs (list, optional): List of input keys for the network.
        """
        self.name = name
        self.architecture = architecture
        self.nr_layers = nr_layers
        self.layer_size = layer_size
        self.activation = activation
        self.config = config
        self.full_freqs = full_freqs
        self.periodic = periodic
        self.adaptive = adaptive
        self.nodes = None
        self.inputs = inputs
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()
        
    def get_activationFunc(self, actFunc: str):
        """
        Retrieves the activation function based on the provided string.

        Args:
            actFunc (str): Name of the activation function.

        Returns:
            Activation: Corresponding activation function.
        """
        activation_functions = {'tanh': Activation.TANH,
                                'relu': Activation.RELU,
                                'selu': Activation.SELU,
                                'leaky': Activation.LEAKY_RELU,
                                'gelu': Activation.GELU,
                                'mish': Activation.MISH,
                                'prelu': Activation.PRELU,
                                'elu': Activation.ELU,
                                'silu': Activation.SILU,
                                'sin': Activation.SIN,
                                'squareplus': Activation.SQUAREPLUS,
                                'softplus': Activation.SOFTPLUS,
                                'stan': Activation.STAN,
                                'identity': Activation.IDENTITY}

        return activation_functions[actFunc]
    
    def get_model(self, model: str):
        """
        Retrieves the model architecture based on the provided string.

        Args:
            model (str): Name of the model architecture.

        Returns:
            class: Corresponding model architecture class.
        """
        models = {'Fourier': FourierNetArch,
                  'FeedForward': FullyConnectedArch,
                  'ModFourier': ModifiedFourierNetArch,
                  'ModFeedForward': ModFullyConnectedArch}

        return models[model]

    def create_network(self, outputs: List[str]):
        """
        Creates a network based on the specified architecture and outputs.

        Args:
            outputs (list): List of output keys for the network.

        Returns:
            model: Instantiated model based on the specified architecture.
        """
        model = self.get_model(self.architecture)
        activation_fn = self.get_activationFunc(self.activation)
        output_keys = [Key(i) for i in outputs]

        input_keys = [Key(inp) for inp in self.inputs]

        if 'FeedForward' in self.architecture:
            periodicity = {"y": (-0.10515748, 0.397283464)} if self.periodic else None

            return model(input_keys=input_keys,
                         output_keys=output_keys,
                         layer_size=self.layer_size,
                         nr_layers=self.nr_layers,
                         activation_fn=activation_fn,
                         adaptive_activations=self.adaptive,
                         periodicity=periodicity)

        else:
            freqs = ('full', [i for i in range(10)]) if self.full_freqs else ("axis", [i for i in range(10)])

            return model(input_keys=input_keys,
                         output_keys=output_keys,
                         layer_size=self.layer_size,
                         nr_layers=self.nr_layers,
                         activation_fn=activation_fn,
                         adaptive_activations=self.adaptive,
                         frequencies=freqs)

    def create_network_nodes(self, viscous: bool):
        """
        Creates network nodes based on the specified configuration and viscosity.

        Args:
            viscous (bool): Whether to use the viscous version of the Navier-Stokes equations.

        Returns:
            list: List of nodes in the network.
        """
        ns = CustomNavierStokes() if viscous else CustomNavierStokes_inviscid()
        navier_stokes_nodes = ns.make_nodes()

        if self.config == 'single':
            flow_net = self.create_network(outputs=['u', 'v', 'rho', 'T'])
            nodes = navier_stokes_nodes + [flow_net.make_node(name=self.name)]

        elif self.config == 'dual':
            flow_net_vel = self.create_network(outputs=['u', 'v'])
            flow_net_comp = self.create_network(outputs=['rho', 'T'])
            nodes = (navier_stokes_nodes
                     + [flow_net_vel.make_node(name=f"{self.name}_vel")]
                     + [flow_net_comp.make_node(name=f"{self.name}_comp")])

        elif self.config == 'total':
            flow_net_u = self.create_network(outputs=['u'])
            flow_net_v = self.create_network(outputs=['v'])
            flow_net_rho = self.create_network(outputs=['rho'])
            flow_net_T = self.create_network(outputs=['T'])
            nodes = (navier_stokes_nodes
                     + [flow_net_u.make_node(name=f"{self.name}_u")]
                     + [flow_net_v.make_node(name=f"{self.name}_v")]
                     + [flow_net_rho.make_node(name=f"{self.name}_rho")]
                     + [flow_net_T.make_node(name=f"{self.name}_T")])
        
        self.nodes = nodes
        return nodes
