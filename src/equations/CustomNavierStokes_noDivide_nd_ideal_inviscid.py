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
        
        T_s = 294.292267
        T_r = 293.549744
        
        rho_s = 2.40159655
        rho_r = 2.3701694
        
        L = 0.0254
        R = 286.7
        a_0 = 30
        c_v = 718
        
        rho_span = rho_s - rho_r
        T_span = T_s - T_r
        
        rho_r_norm = rho_r/rho_span
        T_r_norm = T_r/T_span

        # set equations
        self.equations = {}
        
        self.equations["continuity"] = (
            1/rho_r_norm * ( (rho * u).diff(x) +
                             (rho * v).diff(y) )
            + u.diff(x) 
            + v.diff(y)
        )

        self.equations["momentum_x"] = (
            (a_0**2/(R * T_r)) * (
                2 * rho * u * u.diff(x) +
                u**2 * rho.diff(x) + 
                rho * u * v.diff(y) + 
                rho * v * u.diff(y) + 
                u * v * rho.diff(y)
            ) +
            ((a_0**2 * rho_r_norm)/(R * T_r)) * (
                2 * u * u.diff(x) + 
                u * v.diff(y) + 
                v * u.diff(y)
            ) + 
            (1/T_r_norm) * (
                T * rho.diff(x) + 
                rho * T.diff(x)
            ) +
            rho.diff(x) + 
            (rho_r_norm/T_r_norm) * rho.diff(x) 
        )

        self.equations["momentum_y"] = (
            (a_0**2/(R * T_r)) * (
                2 * rho * v * v.diff(x) +
                v**2 * rho.diff(y) + 
                rho * u * v.diff(x) + 
                rho * v * u.diff(x) + 
                u * v * rho.diff(x)
            ) +
            ((a_0**2 * rho_r_norm)/(R * T_r)) * (
                2 * v * v.diff(y) + 
                u * v.diff(x) + 
                v * u.diff(x)
            ) + 
            (1/T_r_norm) * (
                T * rho.diff(y) + 
                rho * T.diff(y)
            ) +
            rho.diff(y) + 
            (rho_r_norm/T_r_norm) * rho.diff(y) 
        )
            
        self.equations['energy'] = (
             ((1 + R/c_v)/(rho_r_norm * T_r_norm)) * (
                 (rho * u * T).diff(x) + 
                 (rho * v * T).diff(y)
             ) + 
             ((1 + R/c_v)/(rho_r_norm)) * (
                 (u * rho).diff(x) + 
                 (v * rho).diff(y)
             ) +
             (1 + R/c_v) * (
                 u.diff(x) + v.diff(y)
             ) + 
             ((1 + R/c_v)/(T_r_norm)) * (
                 (T * u).diff(x) + 
                 (T * v).diff(y)
             ) + 
             (a_0**2)/(2 * rho_r_norm * T_r_norm * c_v) * (
                 (rho * u * (u**2 + v**2)).diff(x) + 
                 (rho * v * (u**2 + v**2)).diff(y)
             ) + 
             (a_0**2)/(2 * T_r_norm * c_v) * (
                 (u * (u**2 + v**2)).diff(x) + 
                 (v * (u**2 + v**2)).diff(y)
             )
        )