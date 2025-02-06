import pandas as pd
import numpy as np
from typing import Union, Dict, List
import copy
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains
from math import sqrt
from time import time
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import concurrent.futures

from .Network import Network

class Geometry:
    """
    A class to represent geometrical configurations and manage dataset attributes.

    Attributes:
        L (float): Length scale.
        rho_s (float): Scale density.
        rho_r (float): Reference density.
        T_s (float): scale temperature.
        T_r (float): Reference temperature.
        a_0 (float): Speed of sound.
        c_v (float): Specific heat capacity.
        turn (float): Turn angle.
        pitch_ref (float): Reference pitch.
        parametric (dict): Parametric settings.
        name (str): Name of the geometry.
        file_ref (str or None): Reference to the file containing dataset.
        dataset (pd.DataFrame): Dataset loaded from the file.
        N (int): Number of data points in the dataset.
        combination_dict (dict): Dictionary of parameter combinations and their indices.
    """

    def __init__(self,
                 material_props: dict,
                 name: str,
                 dataset : pd.DataFrame,
                 parametric: dict):
        """
        Initializes a new instance of the Geometry class.

        Args:
            material_props (dict): Material properties for the geometry.
            name (str): Name of the geometry.
            file_ref (Union[str, None]): Reference to the file containing dataset.
            parametric (dict): Parametric settings for the geometry.
        """
        self.L = material_props["L"]
        self.rho_s = material_props["rho_s"]
        self.rho_r = material_props["rho_r"]
        self.T_s = material_props["T_s"]
        self.T_r = material_props["T_r"]
        self.a_0 = material_props["a_0"]
        self.c_v = material_props["c_v"]
        self.turning_angle_ref = material_props["turning_angle"]
        self.pitch_ref = material_props["pitch"]
        self.angle_in_ref = material_props["angle_in"]
        self.y_t_ref = material_props["y_t"]
        self.y_mid_ref = material_props["y_mid"]
        self.theta_ref = material_props['theta']

        self.dataset = dataset
        self.N = self.dataset.shape[0]
        self.parametric = parametric
        self.name = name
        
        self.dataset_to_attribute(normalize=True)

    @classmethod
    def from_file(cls, 
                  material_props : Dict[str, float], 
                  file_ref : str, 
                  name: str,
                  parametric: dict): 
        
        dataset = pd.read_csv(file_ref, engine="pyarrow")

        return cls(material_props = material_props,
                   dataset = dataset,
                   name = name,
                   parametric = parametric)
        
    def create_combination_dict(self):
        """
        Creates a dictionary of parameter combinations and their indices.
        """
        param_keys = list(self.parametric.keys())
        sorted_param_keys = sorted(param_keys)
        combination_dict = {}
        current_combination = None
        start_index = 0
        estimated_interval = None

        dataset_array = self.dataset[param_keys].to_numpy()

        i = 0
        while i < self.N:
            row_combination = tuple(dataset_array[i][param_keys.index(key)] for key in sorted_param_keys)

            if row_combination != current_combination:
                if current_combination is not None:
                    combination_dict[current_combination] = (start_index, i - 1)
                    if estimated_interval is None:
                        estimated_interval = i - start_index
                    else:
                        estimated_interval = (estimated_interval + (i - start_index)) // 2

                current_combination = row_combination
                start_index = i

                if estimated_interval is not None:
                    low = max(i + int(estimated_interval * 0.85), i + 1)
                    high = min(i + int(estimated_interval * 1.15), self.N)
                    while low < high:
                        mid = (low + high) // 2
                        mid_combination = tuple(dataset_array[mid][param_keys.index(key)] for key in sorted_param_keys)
                        if mid_combination == current_combination:
                            low = mid + 1
                        else:
                            high = mid
                    i = high
                else:
                    i += 1
            else:
                i += 1

        if current_combination is not None:
            combination_dict[current_combination] = (start_index, self.N - 1)

        self.combination_dict = combination_dict

    def dataset_to_attribute(self, normalize : bool =False):
        """
        Converts dataset columns to class attributes.

        Args:
            normalize (bool): Whether to normalize the attributes.
        """
        for header in list(self.dataset):
            if 'Z [ m ]' in header and 'Normal' not in header:
                if normalize:
                    self.dataset[header] = self.dataset[header].to_numpy() / self.L
                self.x = self.dataset[header].to_numpy().reshape(self.N, 1)
            
            elif 'Y [ m ]' in header and 'Normal' not in header:
                if normalize:
                    self.dataset[header] = self.dataset[header].to_numpy() / self.L
                self.y = self.dataset[header].to_numpy().reshape(self.N, 1)
                
            elif 'Velocity w [ m s^-1 ]' in header or ' Velocity u [ m s^-1 ]' in header or header == 'u':
                if normalize:
                    self.dataset[header] = self.dataset[header].to_numpy() / self.a_0
                self.u = self.dataset[header].to_numpy().reshape(self.N, 1)

            elif ' Velocity v [ m s^-1 ]' in header or header == 'v':
                if normalize:
                    self.dataset[header] = self.dataset[header].to_numpy() / self.a_0
                self.v = self.dataset[header].to_numpy().reshape(self.N, 1)

            elif ' Density [ kg m^-3 ]' in header or header == 'rho':
                if normalize:
                    self.dataset[header] = (self.dataset[header].to_numpy() - self.rho_r) / (self.rho_s - self.rho_r)
                self.rho = self.dataset[header].to_numpy().reshape(self.N, 1)

            elif 'Temperature [ K ]' in header or header == 'T':
                if normalize:
                    self.dataset[header] = (self.dataset[header].to_numpy() - self.T_r) / (self.T_s - self.T_r)
                self.T = self.dataset[header].to_numpy().reshape(self.N, 1)

            elif 'Normal X' in header:
                if normalize:
                    self.dataset[header] = self.dataset[header].to_numpy() / self.L
                self.dx = self.dataset[header].to_numpy().reshape(self.N, 1) 
                
            elif 'Normal Y' in header:
                if normalize:
                    self.dataset[header] = self.dataset[header].to_numpy() / self.L
                self.dy = self.dataset[header].to_numpy().reshape(self.N, 1)

            elif 'sdf' in header:
                if normalize:
                    self.dataset[header] = np.abs(self.dataset[header].to_numpy() / self.L)
                self.sdf = self.dataset[header].to_numpy().reshape(self.N, 1) 

            elif 'turning_angle' in header or 'Turning Angle' in header:
                if normalize:
                    self.dataset[header] = self.dataset[header].to_numpy() / self.turning_angle_ref
                self.turning_angle = self.dataset[header].to_numpy().reshape(self.N, 1)
                    
            elif 'pitch' in header:
                if normalize:
                    self.dataset[header] = self.dataset[header].to_numpy() / self.pitch_ref
                self.pitch = self.dataset[header].to_numpy().reshape(self.N, 1)
                    
            elif 'y_t' in header: 
                if normalize:
                    self.dataset[header] = self.dataset[header].to_numpy() / self.y_t_ref
                self.yt = self.dataset[header].to_numpy().reshape(self.N, 1)
            
            elif 'y_mid' in header: 
                if normalize:
                    self.dataset[header].to_numpy() / self.y_mid_ref
                self.ymid = self.dataset[header].to_numpy().reshape(self.N, 1)

            elif 'theta' in header:
                if normalize:
                    self.dataset[header] = self.dataset[header].to_numpy() / self.theta_ref
                self.theta = self.dataset[header].to_numpy().reshape(self.N, 1)
            
            elif 'angle_in' in header:
                if normalize:
                    self.dataset[header] = self.dataset[header].to_numpy() / self.angle_in_ref
                self.angle_in = self.dataset[header].to_numpy().reshape(self.N, 1)
                
        self.continuity = np.zeros((self.N, 1))
        self.energy = np.zeros((self.N, 1))
        self.momentum_x = np.zeros((self.N, 1))
        self.momentum_y = np.zeros((self.N, 1))
        
    def get_combination(self, parameters: Dict[str, float], create_copy : bool = True):
        """
        Retrieves a subset of the dataset based on a parameter combination.

        Args:
            combination (tuple): Parameter combination.

        Returns:
            Geometry: A subset of the geometry with the specified combination.
        """

        if len(parameters.keys()) == 0:
            raise RuntimeError("No associated parameters")
        
        try:
            combination = tuple([parameters[key] for key in parameters.keys()])
        except:
            print([parameters[key] for key in parameters.keys()])

        idx_first, idx_last = self.combination_dict[combination]
        indices = [i for i in range(idx_first, idx_last + 1)]

        if create_copy:
            return self.subset_select(indices=indices, create_copy= True, reset_index=False)
        else:
            return indices
        
    def assign_network(self, network : Network):
        """
        Assigns a network to the geometry.

        Args:
            network: Network object.
        """
        self.network = network

    def assign_interface(self, networks: List[Network]):
        """
        Assigns an interface to the geometry.

        Args:
            networks: List of networks.
        """
        self.interface = networks
        
    def get_arrayAttributes(self):
        """
        Retrieves the names of attributes that are NumPy arrays.

        Returns:
            List[str]: List of attribute names that are NumPy arrays.
        """
        attr_to_change = []
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, np.ndarray) and not attr_name.startswith('__'):
                attr_to_change.append(attr_name)
        return attr_to_change
    
    def parameter_select(self, parametric: Union[Dict[str, List], None]):
        """
        Selects a subset of the dataset based on parametric settings.

        Args:
            parametric (Union[Dict[str, List], None]): Parametric settings.

        Returns:
            Geometry: A subset of the geometry with the specified parametric settings.
        """
        if parametric is not None:
            query_parts = []
            for column, values in parametric.items():
                if isinstance(values, list):
                    query_parts.append(f"`{column}` in {values}")
            query_string = " & ".join(query_parts)
            filtered_df = self.dataset.query(query_string)
            matching_indices = filtered_df.index.tolist()
            return self.subset_select(indices=matching_indices, create_copy=True)
        
    def subset_select(self, 
                      indices: List[int] = None, 
                      name : str =None, 
                      create_copy : bool =True, 
                      reset_index : bool =True):
        """
        Selects a subset of the dataset based on indices.

        Args:
            indices (list or None): List of indices to select.
            name (str or None): Name of the subset.
            create_copy (bool): Whether to create a copy of the subset.
            reset_index (bool): Whether to reset the index of the subset.

        Returns:
            Geometry: A subset of the geometry with the specified indices.
        """

        if create_copy:
            subset = copy.copy(self)
            obj = subset
        else:
            obj = self         
            
        if name is not None:
            obj.name = name
                
        if indices is not None:
            obj.N = len(indices)
            obj.dataset = obj.dataset.loc[indices]
            obj.dataset_to_attribute()
            if reset_index:
                obj.dataset = obj.dataset.reset_index(drop=True)
                
        return obj 
    
    def trimPassage(self,
                    create_copy : bool =False,
                    reset_index : bool =True,
                    x_min : float = None, 
                    x_max : float = None,
                    y_min : float = None,
                    y_max : float = None,
                    margin: float = 0.005):
        """
        Trims the passage based on specified x and y limits.

        Args:
            create_copy (bool): Whether to create a copy of the trimmed passage.
            reset_index (bool): Whether to reset the index of the trimmed passage.
            x_min (float or None): Minimum x-coordinate.
            x_max (float or None): Maximum x-coordinate.
            y_min (float or None): Minimum y-coordinate.
            y_max (float or None): Maximum y-coordinate.
            margin (float): Margin for trimming.

        Returns:
            Geometry: A trimmed subset of the geometry.
        """
        if (x_min is None and
            x_max is None and
            y_min is None and
            y_max is None):
            
            return self.subset_select(indices=None,
                                    create_copy=False)
                                        
        # Initialize condition array with True values
        cond = np.ones(self.x.shape, dtype=bool)

        # Apply conditions in a vectorized manner
        if x_min is not None:
            cond &= (self.x >= x_min - margin)
        if x_max is not None:
            cond &= (self.x <= x_max + margin)
        if y_min is not None:
            cond &= (self.y >= y_min - margin)
        if y_max is not None:
            cond &= (self.y <= y_max + margin)
        
        # Get indices where condition is True
        ind = np.where(cond)[0]
        indices = self.dataset.index[ind]
                
        geo = self.subset_select(indices=indices, 
                                create_copy=create_copy,
                                reset_index=reset_index)

        return geo
            
class Region(Geometry):
    
    def split(self,
              boundary: Geometry,
              interior_name: str,
              exterior_name: str,
              reset_index: bool = True):
        """
        Splits the region into interior and exterior based on a boundary.

        Args:
            boundary (Geometry): Boundary geometry.
            interior_name (str): Name for the interior region.
            exterior_name (str): Name for the exterior region.
            reset_index (bool): Whether to reset the index of the subsets.

        Returns:
            tuple: A tuple containing the interior and exterior regions and their indices.
        """
        bound = Polygon(list(zip(boundary.x, boundary.y)))

        # Create an array of points
        pts = np.column_stack((self.x, self.y))

        # Vectorized point-in-polygon test
        mask = contains(bound, pts[:, 0], pts[:, 1])

        # Get indices
        dataset_idx = self.dataset.index.to_numpy()
        interior_dataset_idx = dataset_idx[mask]
        exterior_dataset_idx = dataset_idx[~mask]

        interior = self.subset_select(indices=interior_dataset_idx,
                                      name=interior_name,
                                      reset_index=reset_index)
        
        exterior = self.subset_select(indices=exterior_dataset_idx,
                                      name=exterior_name,
                                      reset_index=reset_index)
                
        return (interior, interior_dataset_idx), (exterior, exterior_dataset_idx)

    def create_bounds(self, x_min : float, x_max: float):
        """
        Creates inlet, outlet, top, and bottom bounds for the region.

        Args:
            x_min (float): Minimum x-coordinate.
            x_max (float): Maximum x-coordinate.

        Returns:
            tuple: A tuple containing the inlet, outlet, top, and bottom bounds.
        """
        # Creating Inlet
        indices = np.where(abs(self.x - x_min) < 0.025)[0]
        name = self.name + ' inlet'
        new_inlet = super().subset_select(indices=indices, 
                                          name=name, 
                                          reset_index=False)

        # Creating Outlet
        indices = np.where(abs(self.x - x_max) < 0.025)[0]
        name = self.name + ' outlet'
        new_outlet = super().subset_select(indices=indices, 
                                           name=name, 
                                           reset_index=False)

        n_x = 30
        n_y = 7

        x_min, x_max = min(self.x), max(self.x)
        dx = (x_max - x_min) / n_x
        x = np.linspace(x_min, x_max, n_x)

        unique_combinations = self.dataset[self.parametric.keys()].drop_duplicates()

        indices_max = set()
        indices_min = set()

        iteration = 0
        for _, combination in unique_combinations.iterrows():
            combination_values = combination.values
            combination_dict = dict(zip(self.parametric.keys(), combination_values))

            # Create masks for inlet and outlet datasets
            mask_inlet = (new_inlet.dataset[self.parametric.keys()] == combination_values).all(axis=1)
            mask_outlet = (new_outlet.dataset[self.parametric.keys()] == combination_values).all(axis=1)

            y_inlet_max, y_inlet_min = new_inlet.y[mask_inlet].max(), new_inlet.y[mask_inlet].min()
            y_outlet_max, y_outlet_min = new_outlet.y[mask_outlet].max(), new_outlet.y[mask_outlet].min()
            
            min_max = y_inlet_max if y_inlet_max < y_outlet_max else y_outlet_max
            max_min = y_inlet_min if y_inlet_min > y_outlet_min else y_outlet_min

            y_max = np.linspace(y_inlet_max, y_outlet_max, n_y)
            y_min = np.linspace(y_inlet_min, y_outlet_min, n_y)
            dy = abs(y_inlet_max - y_outlet_max) / n_y
            dist_tol = sqrt((dx / 2) ** 2 + (dy / 2) ** 2)

            # Create a mask for the main dataset
            current_passage = self.get_combination(parameters = combination_dict)
            
            current_passage_top = current_passage.trimPassage(y_min=0.95 * min_max,
                                                              create_copy=True,
                                                              reset_index=False)
            x_current_top = current_passage_top.x
            y_current_top = current_passage_top.y
            
            current_passage_bottom = current_passage.trimPassage(y_max=0.95 * max_min,
                                                                 create_copy=True,
                                                                 reset_index=False)
            x_current_bottom = current_passage_bottom.x
            y_current_bottom = current_passage_bottom.y

            x_found_max = set()
            x_found_min = set()

            found_all_min = False
            found_all_max = False
            found_both = False

            for j in range(n_y):
                if found_both:
                    break

                for i in range(n_x):
                    if found_all_max and found_all_min:
                        found_both = True
                        break

                    if not found_all_max and i not in x_found_max:
                        distances = np.sqrt((x_current_top - x[i]) ** 2 + (y_current_top - y_max[j]) ** 2)
                        ind = np.where(distances < dist_tol)[0]

                        if len(ind) > 0:
                            original_indices = current_passage_top.dataset.index[ind]
                            indices_max.update(original_indices.tolist())
                            x_found_max.add(i)

                            if len(x_found_max) == n_x:
                                found_all_max = True

                    if not found_all_min and i not in x_found_min:
                        distances = np.sqrt((x_current_bottom - x[i]) ** 2 + (y_current_bottom - y_min[-(j + 1)]) ** 2)
                        ind = np.where(distances < dist_tol)[0]

                        if len(ind) > 0:
                            original_indices = current_passage_bottom.dataset.index[ind]
                            indices_min.update(original_indices.tolist())
                            x_found_min.add(i)

                            if len(x_found_min) == n_x:
                                found_all_min = True

            iteration += 1

        name = self.name + ' top'
        new_top = super().subset_select(indices=list(indices_max), name=name, reset_index=False)

        name = self.name + ' bottom'
        new_bottom = super().subset_select(indices=list(indices_min), name=name, reset_index=False)

        return new_inlet, new_outlet, new_top, new_bottom


class Boundary(Geometry):
    
    def extend_boundary(self, name : str, percent: float):
        """
        Extends the boundary by a specified percentage.

        Args:
            name (str): Name for the extended boundary.
            percent (float): Percentage to extend the boundary.

        Returns:
            Boundary: A new boundary extended by the specified percentage.
        """
        new_bounds = super().subset_select(name=name, create_copy=True)
    
        if hasattr(self, 'dx') and hasattr(self, 'dy'):
            new_bounds.x = self.x - self.dx * 0.01 * 0.01 * percent
            new_bounds.y = self.y - self.dy * 0.01 * 0.01 * percent
        else:
            raise NotImplementedError("Boundary does not have normal vectors associated with it")
            
        attr_to_change = new_bounds.get_arrayAttributes()
        attr_to_change.remove('x')
        attr_to_change.remove('y')
        
        for parameter in self.parametric.keys():
            attr_to_change.remove(parameter)
        
        for attr in attr_to_change:
            delattr(new_bounds, attr)
                        
        return new_bounds
