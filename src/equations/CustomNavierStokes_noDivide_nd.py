# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Equations related to Navier Stokes Equations
"""

from sympy import Symbol, Function, Number

from modulus.sym.eq.pde import PDE
from modulus.sym.node import Node


class CustomNavierStokes(PDE):
    """
    Compressible Navier Stokes equations
    Reference:
    https://turbmodels.larc.nasa.gov/implementrans.html

    Parameters
    ==========
    nu : float, Sympy Symbol/Expr, str
        The kinematic viscosity. If `nu` is a str then it is
        converted to Sympy Function of form `nu(x,y,z,t)`.
        If `nu` is a Sympy Symbol or Expression then this
        is substituted into the equation. This allows for
        variable viscosity.
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressible Navier Stokes. Default is 1.
    dim : int
        Dimension of the Navier Stokes (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the Navier-Stokes equations.

    Examples
    ========
    >>> ns = NavierStokes(nu=0.01, rho=1, dim=2)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: u*u__x + v*u__y + p__x + u__t - 0.01*u__x__x - 0.01*u__y__y
      momentum_y: u*v__x + v*v__y + p__y + v__t - 0.01*v__x__x - 0.01*v__y__y
    >>> ns = NavierStokes(nu='nu', rho=1, dim=2, time=False)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: -nu*u__x__x - nu*u__y__y + u*u__x + v*u__y - 2*nu__x*u__x - nu__y*u__y - nu__y*v__x + p__x
      momentum_y: -nu*v__x__x - nu*v__y__y + u*v__x + v*v__y - nu__x*u__y - nu__x*v__x - 2*nu__y*v__y + p__y
    """

    name = "NavierStokes"

    def __init__(self):
        # set params

        # coordinates
        x, y = Symbol("x"), Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}
        
        # velocity components
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        T = Function("T")(*input_variables)
        rho = Function("rho")(*input_variables)
        p = Function("p")(*input_variables)
        
        P_s = 202915.414
        P_r = 199828.2188
        
        T_s = 294.292267
        T_r = 293.549744
        
        rho_s = 2.40159655
        rho_r = 2.3701694
        
        L = 0.0254
        R = 286.7
        a_0 = 30
        c_v = 718

        # set equations
        self.equations = {}
        
        self.equations["continuity"] = (
            ( (rho + rho_r_norm) * u ).diff(x) +
            ( (rho + rho_r_norm) * v ).diff(y)
        )

        
        a_1 = (P_s - P_r)/(a_0**2 * (rho_s - rho_r))
        
        self.equations["momentum_x"] = (
            ((rho + rho_r_norm) * u * u).diff(x) +
            ((rho + rho_r_norm) * u * v).diff(y) + 
            a_1 *((rho + rho_r_norm)*(T + T_r_norm)).diff(x)
            )

        self.equations["momentum_y"] = (
            rho.diff(y) * v**2 + 
            2 * rho * v * v.diff(y) + 
            2 * v * v.diff(y) +
            rho.diff(x) * u * v + 
            rho * u.diff(x) * v + 
            rho * u * v.diff(x) + 
            u.diff(x) * v + 
            u * v.diff(x) +
            a_1 * p.diff(y)
        )
        
        a_2 = (c_v * (T_s - T_r) / a_0**2) 
        a_3 = ((P_s - P_r)/(a_0**2 * (rho_s - rho_r)))
        T_norm = T_r/(T_s-T_r)
        P_norm = P_r/(P_s-P_r)
        
        self.equations['energy'] = (
            1/2 * ((rho + rho_r) * u * (u + v)**2).diff(x) + 
            1/2 * ((rho + rho_r) * v * (u + v)**2).diff(y) + 
            a_2 * ((rho + rho_r) * u * (T + T_norm)).diff(x) +
            a_2 * ((rho + rho_r) * v * (T + T_norm)).diff(y) + 
            a_3 * (u * p.diff(x) + u.diff(x) * (p + P_norm)) +
            a_3 * (v * p.diff(y) + v.diff(y) * (p + P_norm))
        )