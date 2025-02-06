from .Geometry import Geometry, Region, Boundary
from .Network import Network
from .Pairing import Constraint, Plot

from typing import Union, List, Tuple, Dict
from shapely.geometry import Point, Polygon

import warnings

import numpy as np
import pandas as pd
import torch

from time import time
import datetime
import os
from .create_files import create_conf_file
from .Pairing import Plot, Constraint
import sys
import pickle

import modulus.sym
from modulus.sym import main
from modulus.sym.hydra import ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.graph import Graph
from modulus.sym import quantity
from modulus.sym.distributed import DistributedManager
from modulus.sym.domain.constraint import Constraint as Mod_Constraint

from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from .equations.CustomNavierStokes_noDivide_nd_ideal import CustomNavierStokes
from .equations.CustomNavierStokes_noDivide_nd_ideal_inviscid import CustomNavierStokes as CustomNavierStokes_inviscid

from .Design import Design

from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig

import random

default_material_props = {
    "rho_s": 2.40159655,
    "a_0": 40,
    "c_v": 718,
    "T_s": 294.292267,
    "T_r": 293.549744,
    "rho_r": 2.3701694,
    "L": 0.0254,
    "turning_angle": 30,
    "pitch": 1,
    "y_t": 1,
    "y_mid": 1,
    "theta": 1,
    "angle_in": 1
}

class Problem_Domain:
    
    def __init__(self,
                 name: Union[str, None] = None,
                 material_props: Union[dict, None] = None,
                 x_min: Union[float, int] = None,
                 x_max: Union[float, int] = None,
                 y_min: Union[float, int] = None,
                 y_max: Union[float, int] = None,
                 parametric=None):
        """
        Initializes a new instance of the Problem_Domain class.

        Args:
            name (Union[str, None]): The name of the problem domain.
            material_props (Union[dict, None]): Material properties for the problem domain.
            x_min (Union[float, int]): Minimum x-coordinate.
            x_max (Union[float, int]): Maximum x-coordinate.
            y_min (Union[float, int]): Minimum y-coordinate.
            y_max (Union[float, int]): Maximum y-coordinate.
            parametric: Parametric settings for the problem domain.
        """
        self.name = name
        self.material_props = {}
        self.set_material_props(material_props)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.parametric = parametric
        
        inputs = ["x", "y"]
        
        for key in self.parametric.keys():
            inputs.append(key)
            for i, item in enumerate(self.parametric[key]):
                self.parametric[key][i] = item / self.material_props[key]
                                
        self.inputs = inputs
        self.geometries = []
        self.networks = []
        self.constraints = []
        self.plots = []
        self.assigned_geometries = []
        self.inference_geometries = []
        self.designs = []
        
    def set_material_props(self, 
                           input_mat_props: dict):
        """
        Sets the material properties for the problem domain.

        Args:
            input_mat_props (dict): Input material properties.
        """
        if input_mat_props is None:
            self.material_props = default_material_props
        else:
            mat_props = default_material_props.keys()
            for prop in mat_props:
                if prop in input_mat_props:
                    self.material_props[prop] = input_mat_props[prop]
                else:
                    RuntimeWarning(f"Set {prop} to default value of {default_material_props[prop]}")
                    self.material_props[prop] = default_material_props[prop]
                    
    def get_extrema(self, 
                    x_min, 
                    x_max, 
                    y_min, 
                    y_max, 
                    geo):
        """
        Retrieves the extrema (min and max values) for x and y coordinates.

        Args:
            x_min: Minimum x-coordinate.
            x_max: Maximum x-coordinate.
            y_min: Minimum y-coordinate.
            y_max: Maximum y-coordinate.
            geo: Geometry object.

        Returns:
            Tuple of extrema values (x_min, x_max, y_min, y_max).
        """
        def retrieve_extremum(extremum, name, geo):
            if extremum is None:
                if getattr(self, name) is None:
                    inp = 'x' if name.startswith('x') else 'y'
                    extreme_func = max if name.endswith('max') else min
                    actual_extremum = extreme_func(getattr(geo, inp))
                else:
                    actual_extremum = getattr(self, name)
            else:
                actual_extremum = extremum
            return actual_extremum
        
        x_min = retrieve_extremum(x_min, 'x_min', geo)
        y_min = retrieve_extremum(y_min, 'y_min', geo)
        x_max = retrieve_extremum(x_max, 'x_max', geo)
        y_max = retrieve_extremum(y_max, 'y_max', geo)
        
        return x_min, x_max, y_min, y_max
                          
    def add_plotGeometry(self, 
                         geometry=None, 
                         validator=None, 
                         name=None, 
                         x_min=None,
                         y_min=None,
                         x_max=None,
                         y_max=None):
        """
        Adds a plot geometry to the problem domain.

        Args:
            geometry: Geometry object.
            validator: Validator object.
            name: Name of the plot.
            x_min: Minimum x-coordinate.
            y_min: Minimum y-coordinate.
            x_max: Maximum x-coordinate.
            y_max: Maximum y-coordinate.
        """
        if geometry is None:
            geometry = self.plotRegions

        if validator is None:
            validator = self.validator
                            
        if isinstance(geometry, list):
            trimmed_geometry = []
            extrema = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
            
            for geo in geometry:
                new_x_min, new_x_max, new_y_min, new_y_max = self.get_extrema(x_min, x_max, y_min, y_max, geo)
                
                for value, key in [(new_x_min, "x_min"), (new_x_max, "x_max"), (new_y_min, "y_min"), (new_y_max, "y_max")]:
                    if extrema[key] is None:
                        extrema[key] = value
                    else:
                        if 'min' in key:
                            extrema[key] = extrema[key] if extrema[key] < value else value
                        else:
                            extrema[key] = extrema[key] if extrema[key] > value else value
                
                trim = geo.trimPassage(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, margin=0.1, create_copy=True)
                trimmed_geometry.append(trim)
            
            x_min, x_max = extrema['x_min'], extrema['x_max']
            y_min, y_max = extrema['y_min'], extrema['y_max']

        else:
            trimmed_geometry = geometry.trimPassage(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, margin=0.1, create_copy=True)
            new_x_min, new_x_max, new_y_min, new_y_max = self.get_extrema(x_min, x_max, y_min, y_max, geometry)
        
        plot_constraint = Plot(geometry=trimmed_geometry, inputs=self.inputs, name=name, validator=validator, x_min=new_x_min, x_max=new_x_max, y_min=new_y_min, y_max=new_y_max)
        self.plots.append(plot_constraint)
        
    def set_plottingRegions(self, geometries):
        """
        Sets the plotting regions for the problem domain.

        Args:
            geometries: List of geometries.
        """
        self.plotRegions = geometries

    def assign_interface(self, 
                         geometry: Geometry, 
                         networks):
        """
        Assigns an interface to a geometry.

        Args:
            geometry: Geometry object.
            networks: List of networks.
        """
        geometry.assign_interface(networks)

    def set_validation(self, 
                       validator: Geometry):
        """
        Sets the validation method for the problem domain.

        Args:
            validator: Validator object.
        """
        self.validator = validator
    
    def add_inferenceGeometry(self, 
                              design : Design):
        """Add geometry for inference using a Design object

        Args:
            design (Design): Design object
        """        

        passage, _ = design.get_geometry("passage")
        self.inference_geometries.append(passage)

    def encoding(self, networks):


        encoding_key = {}
        decoding_key = {}

        for i, net in enumerate(networks):
            encoding_key[net.name] = i
            decoding_key[i] = net.name

        self.encoding_key = encoding_key
        self.decoding_key = decoding_key

        return encoding_key, decoding_key
    
    def designs_to_dataset(self, 
	                       designs: List[Design],
	                       encoding_key: dict,
	                       geo_to_network: dict) -> pd.DataFrame:
        """
        Converts a list of design objects into a pandas DataFrame suitable for machine learning tasks.

        Args:
            designs (List[Design]): A list of design objects containing geometries and parameters.
            encoding_key (dict): A dictionary mapping network names to their encoded values.
            geo_to_network (dict): A dictionary mapping geometry names to network names.

        Returns:
            pd.DataFrame: A DataFrame containing the encoded design data.
        """

        # Initialize the data dictionary with empty lists for 'x', 'y', and 'network'
        data = {'x': [], 'y': [], 'network': []}

        # Add empty lists for each parameter in the parametric keys
        for parameter in self.parametric.keys():
            data[parameter] = []

        # Iterate over each design in the designs list
        for design in designs:
            
            # Iterate over each geometry name in the design's geometries
            for geo_name in design.geometries.keys():
                
                # Check if the geometry name is in the geo_to_network dictionary
                if geo_name in geo_to_network.keys():
                    geo, indices = design.geometries[geo_name].values()
                    
                    # Iterate over each key in the data dictionary
                    for key in data.keys():
                        if key == 'network':
                            # Get the network name and its encoded value
                            network_name = geo_to_network[geo_name]
                            encoded_network = encoding_key[network_name]
                            # Extend the 'network' list with the encoded network value
                            data[key].extend([encoded_network for i in range(len(indices))])
                        else:
                            # Get the attribute from the geometry and flatten it
                            inp = getattr(geo, key)[indices].flatten()
                            # Extend the corresponding list in the data dictionary
                            data[key].extend(inp)

        # Convert the data dictionary to a pandas DataFrame and return it
        return pd.DataFrame(data)

    def assign_network(self,
                    associations: List[Tuple[Network, List[Geometry]]]):
        """
        Assigns a network to geometries and trains a KNN model to differentiate between networks.

        Args:
            associations (List[Tuple[Network, List[Geometry]]]): A list of tuples where each tuple contains a network and a list of geometries.
        """
        
        # Initialize lists and dictionaries
        networks = []
        geo_to_network = {}
        
        # Iterate over each association in the associations list
        for association in associations:
            
            network, geometries = association
            networks.append(network)
            
            # Check if geometries is a list
            if isinstance(geometries, list):
                
                # Assign network to each geometry in the list
                for geo in geometries:
                    geo_to_network[geo.name] = network.name
                    geo.assign_network(network)
                    self.assigned_geometries.append(geo)
            
            else:
                # Assign network to a single geometry
                geo_to_network[geometries.name] = network.name
                geometries.assign_network(network)
                self.assigned_geometries.append(geometries)
        
        # Encode the networks
        encoding_key, _ = self.encoding(networks)
        
        # Split designs into training and testing sets
        training_designs = []
        testing_designs = []
        train_test_split = 0.8
        N_designs = len(self.designs)
        training_designs = random.sample(self.designs, round(N_designs * train_test_split))
        testing_designs = list(set(self.designs) - set(training_designs))
        
        # Convert designs to datasets
        training_data = self.designs_to_dataset(designs=training_designs,
                                                encoding_key=encoding_key,
                                                geo_to_network=geo_to_network)
        
        testing_data = self.designs_to_dataset(designs=testing_designs,
                                                encoding_key=encoding_key,
                                                geo_to_network=geo_to_network)
        
        # Define features and target
        features = ['x', 'y']
        for key in self.parametric.keys():
            features.append(key)
        target = 'network'
        
        # Split data into features and target for training and testing
        X_train = training_data[features]
        Y_train = training_data[target]
        X_test = testing_data[features]
        Y_test = testing_data[target]
        
        # Initialize the KNN model
        model = KNeighborsClassifier(n_neighbors=3)
        
        # Train and evaluate the model
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred)
        
        print(f"KNN model accuracy: {accuracy}")
        
        # Store the trained model
        self.network_differentiator = model

    def create_network(self, **kwargs):
        """
        Creates a new network and adds it to the problem domain.

        Args:
            \*\*kwargs: Additional arguments for the network.

        Returns:
            Network object.
        """
        network = Network(inputs=self.inputs, **kwargs)
        self.networks.append(network)
        return network
        
    def add_constraint(self, **kwargs):
        """
        Adds a constraint to the problem domain.

        Args:
            \*\*kwargs: Additional arguments for the constraint.
        """
        constraint = Constraint(parametric=self.parametric, inputs=self.inputs, **kwargs)
        self.constraints.append(constraint)
        
    def create_geometry(self, 
                        geometry_type : Union[Region, Boundary], 
                        file_ref: str,
                        **kwargs):
        """
        Creates a new geometry and adds it to the problem domain.

        Args:
            geometry_type: Type of the geometry.
            \*\*kwargs: Additional arguments for the geometry.

        Returns:
            Geometry object.
        """
        geometry = geometry_type.from_file(material_props=self.material_props, 
                                           file_ref = file_ref,
                                           parametric= self.parametric, 
                                           **kwargs)
        
        geometry.trimPassage(x_min=self.x_min, 
                             x_max=self.x_max, 
                             y_min=self.y_min, 
                             y_max=self.y_max)
        
        geometry.create_combination_dict()
        self.geometries.append(geometry)
        return geometry
    
    def create_boundary(self, file_ref: str, name: str):
        """
        Creates a boundary geometry.

        Args:
            \*\*kwargs: Additional arguments for the boundary.

        Returns:
            Boundary object.
        """
        return self.create_geometry(geometry_type=Boundary, 
                                    file_ref = file_ref, 
                                    name = name)
        
    def create_region(self, file_ref: str, name: str):
        """
        Creates a region geometry.

        Args:
            \*\*kwargs: Additional arguments for the region.

        Returns:
            Region object.
        """
        return self.create_geometry(geometry_type=Region, 
                                    file_ref = file_ref, 
                                    name = name)
    
    def create_bounds(self, geometry : Geometry):
        """
        Creates bounds for a geometry.

        Args:
            geometry: Geometry object.

        Returns:
            Tuple of new inlet, outlet, top, and bottom bounds.
        """
        new_inlet, new_outlet, new_top, new_bottom = geometry.create_bounds(x_min=self.x_min, x_max=self.x_max)
        return new_inlet, new_outlet, new_top, new_bottom
    
    def trimPassage(self, 
                    geometry : Geometry, 
                    x_min : float = None, 
                    x_max : float = None, 
                    y_min : float = None, 
                    y_max : float = None):
        """
        Trims a passage geometry.

        Args:
            geometry: Geometry object.
            x_min: Minimum x-coordinate.
            x_max: Maximum x-coordinate.
            y_min: Minimum y-coordinate.
            y_max: Maximum y-coordinate.
        """
        clone_geo = geometry.subset_select(create_copy = True)
        
        clone_geo.trimPassage(x_min = x_min, 
                             x_max = x_max,
                             y_min = y_min, 
                             y_max = y_max)
        
        return clone_geo
        
    
    def create_designs(self, 
                       blade: Boundary, 
                       passage: Region):
        
        common_combinations = blade.combination_dict.keys() & passage.combination_dict.keys()
        sorted_param_keys = sorted(self.parametric.keys())

        for combination in common_combinations:
            parameters = dict(zip(sorted_param_keys, list(combination)))
            design = Design.from_geometry(parameters = parameters, 
                                          blade = blade,
                                          passage = passage)
            
            self.designs.append(design)

    def generate_design(self,
                        parameters: Dict[str, Tuple[float]]):
        
        x_min, x_max, y_min, y_max = -1, 1, -0.3, 0.3
        
        return Design.from_parameters(parameters = parameters,
                                      material_props = self.material_props,
                                      parametric = self.parametric,
                                      x_min = x_min, 
                                      x_max = x_max, 
                                      y_min = y_min, 
                                      y_max = y_max)
                       
    def divideByBlade(self,
                      blade : Boundary,
                      passage: Region,
                      percent: Union[float, int]):
        """
        Divides a passage by a blade into boundary layer and inviscid region.

        This method separates the passage region into a boundary layer and an inviscid region based on the specified blade and percentage. It identifies unique parameter combinations in both the blade and passage datasets, finds their intersection, and then splits the passage accordingly.

        Args:
            blade: The blade object containing the dataset to be used for division.
            passage: The passage object containing the dataset to be divided.
            percent (Union[float, int]): The percentage of the blade to be used for creating the interface boundary.

        Returns:
            tuple: A tuple containing the interface boundary, boundary layer, and inviscid region.
        """
        BL_indexes, IR_indexes = [], []

        interface = blade.subset_select(name = 'interface',
                                        indices = [i for i in range(blade.N)])  
        
        attr_to_change = interface.get_arrayAttributes()
        attr_to_change.remove('x')
        attr_to_change.remove('y')
        
        for parameter in self.parametric.keys():
            attr_to_change.remove(parameter)
        
        for attr in attr_to_change:
            delattr(interface, attr)
        
        for design in self.designs:

            BL_index, IR_index = design.divideByBlade(percent, 
                                                      interface)
            
            BL_indexes.extend(BL_index)
            IR_indexes.extend(IR_index)
                    
        boundaryLayer = passage.subset_select(indices=BL_indexes, name="boundary_layer", create_copy=True)
        inviscidRegion = passage.subset_select(indices=IR_indexes, name="inviscid_region", create_copy=True)
        
        interface.interface_u = np.zeros((interface.N, 1))
        interface.interface_v = np.zeros((interface.N, 1))
        interface.interface_rho = np.zeros((interface.N, 1))
        interface.interface_T = np.zeros((interface.N, 1))

        return interface, boundaryLayer, inviscidRegion

    def create_run(self, 
                   parent_dir: str, 
                   training: dict, 
                   modulus: dict):
        """
        Creates a run configuration and executes it.

        Args:
            parent_dir: Parent directory for the run.
            training: Training configuration.
            modulus: Modulus configuration.
        """

        # If run has not been created (path does not exist), then create directory
        if not os.path.exists(os.path.join(parent_dir, 'outputs', self.name)):
            
            directory = str(datetime.datetime.now()).replace(':', "-") if self.name is None else self.name
            parent_dir_out = os.path.join(parent_dir, "outputs")
            path = os.path.join(parent_dir_out, directory)
            os.mkdir(path)
            os.chdir(path)

            create_conf_file(path, training, modulus)

            filehandler = open(f"{path}/problem_domain.obj", "wb")
            pickle.dump(self, filehandler)
            filehandler.close()

            os.chdir(path)
        
            current_path = os.getcwd()

        else:
            current_path = os.path.join(parent_dir, 'outputs', self.name)
            os.chdir(current_path)
        
        @main(config_path=fr"{current_path}/conf", config_name="conf")
        def run(cfg: ModulusConfig) -> None:

            # make flow domain
            flow_domain = Domain()
            
            #######################################################################
            #######                Create Nodes for Networks                 ######
            #######################################################################
            
            for network in self.networks:
                network.create_network_nodes(viscous = cfg.custom.viscous)
            
            #######################################################################
            #######                Constraints                               ######
            #######################################################################
            
            for constraint in self.constraints:
                constraint.create_modObj(mod_domain = flow_domain)
                                                                        
            #######################################################################
            #######                PLOTTTING                                 ######
            #######################################################################
            
            for plot in self.plots:
                plot.create_modObj(mod_domain = flow_domain)
                                        
            #######################################################################
            #######                SOLVING                                   ######
            #######################################################################
                    
            # make solver
            flow_slv = Solver(cfg, flow_domain)

            # start flow solver
            flow_slv.solve()

        run()

    def evaluate_run(self, 
                 output_path: str):
        """
        Evaluates the run by creating nodes for networks, applying constraints, 
        performing known and unknown geometry inference, and saving the results to CSV files.

        Args:
            output_path (str): The path to the output directory.
        """
        
        # Change the current working directory to the output path
        os.chdir(output_path)

        @main(config_path=fr"{output_path}/conf", config_name="conf")
        def evaluate(cfg: ModulusConfig) -> None:

            # Make flow domain
            flow_domain = Domain()
            
            #######################################################################
            #######                Create Nodes for Networks                 ######
            #######################################################################
            
            # Create network nodes for each network in self.networks
            for network in self.networks:
                network.create_network_nodes(viscous=cfg.custom.viscous)
            
            #######################################################################
            #######                Constraints                               ######
            #######################################################################
            
            # Create Modulus objects for each constraint in self.constraints
            for constraint in self.constraints:
                constraint.create_modObj(mod_domain=flow_domain)
                                                                        
            #######################################################################
            #######                Known Geometry Inference                  ######
            #######################################################################

            # Initialize distributed manager and get the device
            manager = DistributedManager()
            device = manager.device

            # Initialize a dictionary to store full data
            full_data = {}

            # Initialize lists for each input in full_data
            for inp in self.inputs:
                full_data[inp] = []
            
            # Initialize lists for each output in full_data
            for output in ["u", "v", "rho", "T"]:
                full_data[output] = []
            
            # Iterate over each assigned geometry
            for geo in self.assigned_geometries:
                # Check if the network has a graph attribute, if not, create one
                if not hasattr(geo.network, 'graph'):
                    net_graph = Graph(geo.network.nodes, 
                                    Key.convert_list(self.inputs), 
                                    Key.convert_list(["u", "v", "rho", "T"]))
                    geo.network.graph = net_graph
                
                # Move the network graph to the device
                geo.network.graph.to(device)

                # Initialize a dictionary to store input variables
                in_vars = {}

                # Convert each input to a tensor and move to CUDA
                for inp in self.inputs:
                    in_vars[inp] = torch.tensor(getattr(geo, inp)).cuda()

                # Set the device for input variables
                invar = Mod_Constraint._set_device(
                    in_vars, device=device, requires_grad=False
                )

                # Get the output variables from the network graph
                outvar = geo.network.graph(invar)

                # Initialize a dictionary to store data
                data = {}

                # Convert input variables to numpy arrays and store in data
                for key, value in invar.items():
                    data[key] = value[:, 0].detach().cpu().numpy().reshape(-1, 1).flatten()

                # Extend full_data with the data
                for key in data.keys():
                    full_data[key].extend(data[key])
                
                # Create a DataFrame from data
                df = pd.DataFrame(data)

                # Save the DataFrame to a CSV file
                df.to_csv(f"{geo.name}_eval.csv", index=False)
            
            # Create a DataFrame from full_data
            df_full = pd.DataFrame(full_data)

            # Save the DataFrame to a CSV file
            df_full.to_csv(f"full_eval.csv", index=False)

            #######################################################################
            #######                Unknown Geometry Inference                ######
            #######################################################################

            # Initialize distributed manager and get the device
            manager = DistributedManager()
            device = manager.device

            # Initialize a dictionary to store separated networks
            N_networks = len(self.networks)
            separated_networks = {}
            for i in range(N_networks):
                key = self.decoding_key[i]
                separated_networks[key] = None

            # Iterate over each inference geometry
            for geo in self.inference_geometries:

                # Initialize a dictionary to store input variables for inference
                in_vars_inf = {}

                # Convert each input to a numpy array and store in in_vars_inf
                for inp in self.inputs:
                    in_vars_inf[inp] = torch.tensor(getattr(geo, inp)).cpu().numpy().reshape(1, -1)[0]

                # Create a DataFrame from in_vars_inf
                new_data = pd.DataFrame(in_vars_inf)

                # Predict the network association using the network differentiator
                network_association = self.network_differentiator.predict(new_data)

                # Separate the data based on network association
                for i in range(N_networks):
                    indices = np.where(network_association == i)[0]
                    subset = new_data.loc[indices]
                    key = self.decoding_key[i]
                    if separated_networks[key] is None:
                        separated_networks[key] = subset
                    else:
                        separated_networks[key] = pd.concat([separated_networks[key], subset], ignore_index=True)

            # Initialize a dictionary to store full predictions
            full_pred = {}

            # Iterate over each network
            for network in self.networks:
                
                # Move the network graph to the device
                network.graph.to(device)

                # Convert separated networks to a dictionary of lists
                in_vars_inf = separated_networks[network.name].to_dict('list')

                # Convert each input to a tensor and move to CUDA
                for inp in self.inputs:
                    column_vec = np.array(in_vars_inf[inp]).reshape(-1, 1)
                    tensor_form = torch.tensor(column_vec)
                    in_vars_inf[inp] = tensor_form.to(device).cuda()

                # Set the device for input variables for inference
                invar_inf = Mod_Constraint._set_device(
                    in_vars_inf, device=device, requires_grad=False
                )

                # Get the output variables from the network graph
                outvar_inf = network.graph(invar_inf)

                # Convert input variables to numpy arrays and store in full_pred
                for key, value in invar_inf.items():
                    if key not in full_pred:
                        full_pred[key] = value[:, 0].detach().cpu().numpy().reshape(-1, 1).flatten()
                    else:
                        input_values = value[:, 0].detach().cpu().numpy().reshape(-1, 1).flatten()
                        full_pred[key] = np.concatenate((full_pred[key], input_values))
                
            # Create a DataFrame from full_pred
            full_pred_df = pd.DataFrame(full_pred)

            # Save the DataFrame to a CSV file
            full_pred_df.to_csv(f"inference_geometries_eval.csv", index=False)

            #######################################################################
            #######                Evaluating                                ######
            #######################################################################
                    
            # Make solver
            flow_slv = Solver(cfg, flow_domain)

            # Start flow solver
            flow_slv.eval()

        # Run the evaluate function
        evaluate()


    @staticmethod
    def load(parent_dir: str, 
            name: str):
        """
        Loads a problem domain object from a specified directory.

        Args:
            parent_dir (str): The parent directory where the outputs are stored.
            name (str): The name of the specific output directory to load from.

        Returns:
            The loaded problem domain object.

        Raises:
            FileNotFoundError: If the specified path does not exist.
        """
        
        # Construct the path to the problem domain object
        path = os.path.join(parent_dir, 'outputs', name, 'problem_domain.obj')

        # Check if the path exists
        if os.path.exists(path):
            # Open the file in binary read mode
            with open(path, 'rb') as file:
                # Load the problem domain object using pickle
                prob_domain = pickle.load(file)
            
            # Return the loaded problem domain object
            return prob_domain
        
        else:
            # Raise an error if the path does not exist
            raise FileNotFoundError(f"The specified path {path} does not exist.")
