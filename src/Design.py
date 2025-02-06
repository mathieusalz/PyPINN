import numpy as np
import pandas as pd

from .Geometry import Geometry, Region, Boundary
from typing import List, Dict, Any

from shapely.geometry import Point, Polygon
from shapely.vectorized import contains

class Design:

    def __init__(self, 
                 parameters: Dict[str, float],
                 geometries: Dict[str, Dict[str, Any]]):
        """
        Initializes a Design object with parameters and geometries.

        Args:
            parameters (Dict[str, float]): A dictionary of design parameters.
            geometries (Dict[str, Dict[str, Any]]): A dictionary of geometries.
        """
        self.parameters = parameters
        self.geometries = geometries 
        
    @staticmethod
    def order_blade_points(geometry: Geometry, indices: List[int]):
        """
        Orders the blade points based on their x and y coordinates.

        Args:
            geometry: The geometry object containing the dataset.
            indices: The indices of the points to be ordered.

        Returns:
            np.ndarray: An array of ordered points.
        """
        subset_dataset = geometry.dataset.iloc[indices]

        current_airfoil_x = subset_dataset[' Z [ m ]'].to_numpy()
        current_airfoil_y = subset_dataset[' Y [ m ]'].to_numpy()

        coordinates = np.column_stack((current_airfoil_x, current_airfoil_y))

        # Find the leading edge (point with the smallest x-coordinate)
        leading_edge = coordinates[np.argmin(coordinates[:, 0])]
        # Find the trailing edge (point with the largest x-coordinate)
        trailing_edge = coordinates[np.argmax(coordinates[:, 0])]

        # Sort all points by x-coordinate
        sorted_coordinates = coordinates[np.argsort(coordinates[:, 0])]

        # Initialize the top half list with the leading edge
        top_half = [leading_edge]
        current_point = leading_edge

        # Remove the leading edge from the sorted coordinates
        sorted_coordinates = sorted_coordinates[sorted_coordinates[:, 0] != leading_edge[0]]

        distances = np.linalg.norm(sorted_coordinates - current_point, axis=1)
        
        # Find the two closest points
        closest_indices = np.argsort(distances)[:2]
        closest_points = sorted_coordinates[closest_indices]
        
        # Choose the point with the higher y-coordinate
        if closest_points[0][1] > closest_points[1][1]:
            chosen_point = closest_points[0]
            chosen_index = closest_indices[0]
        else:
            chosen_point = closest_points[1]
            chosen_index = closest_indices[1]
        
        # Add the chosen point to the top half list
        top_half.append(chosen_point)
        
        # Remove the chosen point from the sorted coordinates
        sorted_coordinates = np.delete(sorted_coordinates, chosen_index, axis=0)
        
        # Update the current point
        current_point = chosen_point

        # Construct the top half
        while not np.array_equal(current_point, trailing_edge):
            # Find the closest point in terms of x and y coordinates
            distances = np.linalg.norm(sorted_coordinates - current_point, axis=1)
            closest_index = np.argmin(distances)
            closest_point = sorted_coordinates[closest_index]
            
            # Add the closest point to the top half list
            top_half.append(closest_point)
            
            # Remove the closest point from the sorted coordinates
            sorted_coordinates = np.delete(sorted_coordinates, closest_index, axis=0)
            
            # Update the current point
            current_point = closest_point

        # The remaining points form the bottom half
        bottom_half = sorted_coordinates

        # Sort top and bottom points by x-coordinate
        top_half = np.array(top_half)
        bottom_half = np.array(bottom_half)
        top_half = top_half[np.argsort(top_half[:, 0])]
        bottom_half = bottom_half[np.argsort(bottom_half[:, 0])]

        # Concatenate points: top points + reversed bottom points
        ordered_points = np.concatenate((top_half, bottom_half[::-1]))

        return ordered_points

    @staticmethod
    def reorder_blade_dataset(geometry: Geometry, indices: List[int]):
        """
        Reorders the blade dataset based on the ordered points.

        Args:
            geometry: The geometry object containing the dataset.
            indices: The indices of the points to be reordered.
        """
        # Get the ordered points and their indices
        ordered_points = Design.order_blade_points(geometry, indices)
        swapped_orderd_points = np.column_stack((ordered_points[:,1], ordered_points[:,0]))
        
        # Create a DataFrame from the ordered points
        ordered_df = pd.DataFrame(swapped_orderd_points, columns=[' Y [ m ]', ' Z [ m ]'])
        
        # Merge with the original dataset to reorder the relevant rows
        reordered_relevant_rows = pd.merge(ordered_df, geometry.dataset, on=[' Y [ m ]', ' Z [ m ]'], how='left')

        geometry.dataset.iloc[indices] = reordered_relevant_rows.iloc[[i for i in range(reordered_relevant_rows.shape[0])]]

        geometry.dataset_to_attribute(normalize = False)

    @staticmethod
    def calculate_centroid_triangle(x: np.ndarray, y: np.ndarray):
        """
        Calculates the centroid of a triangle.

        Args:
            x: The x-coordinates of the triangle vertices.
            y: The y-coordinates of the triangle vertices.

        Returns:
            tuple: The centroid coordinates.
        """
        polygon = np.column_stack((x.reshape(-1,1), y.reshape(-1,1)))

        # Same polygon, but with vertices cycled around. Now the polygon
        # decomposes into triangles of the form origin-polygon[i]-polygon2[i]
        polygon2 = np.roll(polygon, -1, axis=0)

        # Compute signed area of each triangle
        signed_areas = 0.5 * np.cross(polygon, polygon2)

        # Compute centroid of each triangle
        centroids = (polygon + polygon2) / 3.0

        # Get average of those centroids, weighted by the signed areas.
        centroid = np.average(centroids, axis=0, weights=signed_areas)

        return centroid
    
    @staticmethod
    def calculate_centroid_mean(x: np.ndarray, y: np.ndarray):
        """
        Calculates the centroid of a set of points using the mean method.

        Args:
            x: The x-coordinates of the points.
            y: The y-coordinates of the points.

        Returns:
            tuple: The centroid coordinates.
        """
        centroid = (np.mean(x), np.mean(y))
        return centroid
    
    @staticmethod
    def calculate_centroid_shapely(x: np.ndarray, y: np.ndarray):
        """
        Calculates the centroid of a polygon using the Shapely library.

        Args:
            x: The x-coordinates of the polygon vertices.
            y: The y-coordinates of the polygon vertices.

        Returns:
            tuple: The centroid coordinates.
        """
        polygon = Polygon(zip(x, y))
        centroid = polygon.centroid
        return (centroid.x, centroid.y)


    @classmethod
    def from_geometry(cls,
                    parameters: Dict[str, float], 
                    blade: Geometry, 
                    passage: Geometry):
        """
        Creates a Design object from existing geometry objects.

        Args:
            parameters (Dict[str, float]): A dictionary of design parameters.
            blade (Geometry): The blade geometry object.
            passage (Geometry): The passage geometry object.

        Returns:
            Design: A new Design object.
        """
        # Get the indices of the blade and passage geometries based on the parameters
        blade_indices = blade.get_combination(parameters, create_copy=False)
        passage_indices = passage.get_combination(parameters, create_copy=False)
        
        # Reorder the blade dataset based on the ordered points
        Design.reorder_blade_dataset(blade, blade_indices)
        
        # Calculate the centroid of the blade geometry
        centroid = Design.calculate_centroid_shapely(blade.x[blade_indices], blade.y[blade_indices])

        # Adjust the blade dataset coordinates to center around the centroid
        blade.dataset.loc[blade_indices, ' Z [ m ]'] = np.array([x - centroid[0] for x in blade.x[blade_indices]]).flatten() 
        blade.dataset.loc[blade_indices, ' Y [ m ]'] = np.array([y - centroid[1] for y in blade.y[blade_indices]]).flatten()

        # Update the blade dataset attributes
        blade.dataset_to_attribute(normalize=False)

        # Adjust the passage dataset coordinates to center around the centroid
        passage.dataset.loc[passage_indices, ' Z [ m ]'] = np.array([x - centroid[0] for x in passage.x[passage_indices]]).flatten() 
        passage.dataset.loc[passage_indices, ' Y [ m ]'] = np.array([y - centroid[1] for y in passage.y[passage_indices]]).flatten() 

        # Update the passage dataset attributes
        passage.dataset_to_attribute(normalize=False)

        # Create a dictionary to store the geometries
        geometries = {
            'blade': {"geometry": blade, "indices": blade_indices},
            'passage': {"geometry": passage, "indices": passage_indices}
        }

        # Return a new Design object
        return cls(parameters=parameters, geometries=geometries)

    @classmethod
    def from_parameters(cls, 
                        material_props: Dict[str, float],
                        parametric: dict,
                        parameters: Dict[str, float],
                        x_min: float, x_max: float, y_min: float, y_max: float):
        """
        Creates a Design object from given parameters.

        Args:
            material_props: Material properties.
            parametric: Parametric information.
            parameters: Design parameters.
            x_min: Minimum x-coordinate.
            x_max: Maximum x-coordinate.
            y_min: Minimum y-coordinate.
            y_max: Maximum y-coordinate.

        Returns:
            Design: A new Design object.
        """
        raise NotImplementedError

        def generate_passage(blade: Boundary) -> Region:
            """
            Generates the passage geometry.

            Args:
                blade (Boundary): The blade boundary object.

            Returns:
                Region: The passage region object.
            """
            # Extract the blade coordinates and create a polygon
            points = np.column_stack((blade.x, blade.y))
            wrapted_points = np.vstack((points, points[0]))
            bound = Polygon(wrapted_points)

            # Create an array of points within the specified x and y ranges
            x = np.linspace(x_min, x_max, 50)
            y = np.linspace(y_min, y_max, 50)
            X, Y = np.meshgrid(x, y)
            pts = np.column_stack((X.flatten(), Y.flatten()))

            # Perform a vectorized point-in-polygon test
            mask = contains(bound, pts[:, 0], pts[:, 1])

            # Extract the exterior points
            exterior = pts[~mask]
            exterior_x, exterior_y = exterior[:, 0], exterior[:, 1]

            # Create a dataset with the exterior points and turning angle
            data = {
                ' Z [ m ]': exterior_x * material_props['L'],
                ' Y [ m ]': exterior_y * material_props['L'],
                'turning_angle': turning_angle
            }
            
            dataset = pd.DataFrame(data)

            # Create a Region object for the passage
            passage = Region(
                material_props=material_props,
                dataset=dataset,
                name='passage',
                parametric=parametric
            )
            
            return passage
        
        # Generate the blade and passage geometries
        blade = generate_blade()
        passage = generate_passage(blade)

        # Create a dictionary to store the geometries
        geometries = {
            'blade': {"geometry": blade, "indices": [i for i in range(blade.N)]},
            'passage': {"geometry": passage, "indices": [i for i in range(passage.N)]}
        }
        
        # Return a new Design object
        return cls(parameters=parameters, geometries=geometries)
        
    def add_geometry(self, name: str, geo: Geometry, indices: List[int]):
        """
        Adds a geometry to the design.

        Args:
            name (str): The name of the geometry.
            geo (Geometry): The geometry object to be added.
            indices (list): The indices of the geometry points.
        """
        # Add the geometry and its indices to the geometries dictionary
        self.geometries[name] = {'geometry': geo, 'indices': indices}

    def get_geometry(self, name:str):
        """
        Retrieves a geometry from the design.

        Args:
            name (str): The name of the geometry to retrieve.

        Returns:
            tuple: The geometry object and its indices.
        """
        # Return the geometry and its indices from the geometries dictionary
        return (self.geometries[name]['geometry'], self.geometries[name]['indices'])

    def get_blade(self):
        """
        Retrieves the blade geometry and its coordinates.

        Returns:
            tuple: The x and y coordinates of the blade points.
        """
        # Get the blade geometry and its indices
        blade = self.geometries['blade']['geometry']
        indices = self.geometries['blade']['indices']
        # Return the x and y coordinates of the blade points
        return (blade.x[indices], blade.y[indices])

    def get_passage(self):
        """
        Retrieves the passage geometry and its coordinates.

        Returns:
            tuple: The x and y coordinates of the passage points.
        """
        # Get the passage geometry and its indices
        passage = self.geometries['passage']['geometry']
        indices = self.geometries['passage']['indices']
        # Return the x and y coordinates of the passage points
        return (passage.x[indices], passage.y[indices])

    def get_normals(self):
        """
        Calculates and retrieves the normals of the blade geometry.

        Returns:
            tuple: The x and y components of the normals.
        """
        # Check if the normals have already been calculated
        if hasattr(self, 'blade_dx'):
            # Return the pre-calculated normals
            return self.blade_dx, self.blade_dy
        
        # Get the blade coordinates
        x, y = self.get_blade()
        airfoil_points = np.column_stack((x, y))
        
        # Wrap the points to form a closed loop
        wrapted_points = np.vstack((airfoil_points, airfoil_points[0]))

        normal_x = []
        normal_y = []
        areas = []

        # Calculate the normals for each segment of the blade
        for v1, v2 in zip(wrapted_points[:-1], wrapted_points[1:]):
            # Calculate the segment length
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            area = (dx**2 + dy**2) ** 0.5
            areas.append(area)

            # Generate the normals
            normal_x.append(-dy / area)
            normal_y.append(dx / area)
            
        # Convert the normals to numpy arrays
        normal_x = np.array(normal_x).reshape(-1, 1)
        normal_y = np.array(normal_y).reshape(-1, 1)

        # Store the normals in the object
        self.blade_dx = normal_x
        self.blade_dy = normal_y

        # Return the normals
        return normal_x, normal_y

    def extend_interface_boundary(self, percent: float):
        """
        Extends the interface boundary by a given percentage.

        Args:
            percent (float): The percentage by which to extend the boundary.
        """
        # Get the normals and blade coordinates
        normal_x, normal_y = self.get_normals()
        x, y = self.get_blade()

        # Calculate the new boundary coordinates
        new_bounds_x = np.add(x, normal_x * 0.01 * percent)
        new_bounds_y = np.add(y, normal_y * 0.01 * percent)

        # Get the interface geometry and its indices
        interface, indices = self.get_geometry(name='interface')

        # Update the interface coordinates
        interface.x[indices] = new_bounds_x.reshape(-1, 1)
        interface.y[indices] = new_bounds_y.reshape(-1, 1)

    def split(self):
        """
        Splits the passage geometry into boundary layer and inviscid region based on the interface.

        Returns:
            tuple: The indices of the boundary layer and inviscid region points.
        """
        # Get the interface geometry and its indices
        interface, interface_indices = self.get_geometry(name='interface')
        # Create a polygon from the interface points
        bound = Polygon(list(zip(interface.x[interface_indices], interface.y[interface_indices])))

        # Get the passage geometry and its indices
        passage, passage_indices = self.get_geometry(name='passage')
        pts = np.column_stack((passage.x[passage_indices], passage.y[passage_indices]))

        # Perform a vectorized point-in-polygon test
        mask = contains(bound, pts[:, 0], pts[:, 1])

        # Get the indices of the boundary layer and inviscid region points
        bl_idx = np.array(passage_indices)[mask]
        ir_idx = np.array(passage_indices)[~mask]
                
        return bl_idx, ir_idx        

    def divideByBlade(self, 
                    percent: float, 
                    interface: Boundary):
        """
        Divides the passage geometry into boundary layer and inviscid region based on the blade.

        Args:
            percent (float): The percentage by which to extend the interface boundary.
            interface (Boundary): The interface boundary object.

        Returns:
            tuple: The indices of the boundary layer and inviscid region points.
        """
        # Get the interface indices based on the parameters
        interface_indices = interface.get_combination(parameters=self.parameters, create_copy=False)

        # Add the interface geometry to the design
        self.add_geometry(name='interface', geo=interface, indices=interface_indices)

        # Extend the interface boundary
        self.extend_interface_boundary(percent)

        # Split the passage geometry into boundary layer and inviscid region
        bl_idx, ir_idx = self.split()

        # Get the passage geometry
        passage, _ = self.get_geometry('passage')

        # Add the boundary layer and inviscid region geometries to the design
        self.add_geometry(name='boundary_layer', geo=passage, indices=bl_idx)
        self.add_geometry(name='inviscid_region', geo=passage, indices=ir_idx)

        return bl_idx, ir_idx

