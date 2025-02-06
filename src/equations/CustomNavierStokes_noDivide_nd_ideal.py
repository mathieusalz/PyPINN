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
        nu = 1.57e-5
        k = 0.0251
        
        rho_span = rho_s - rho_r
        T_span = T_s - T_r
        
        rho_r_norm = rho_r/rho_span
        T_r_norm = T_r/T_span
        
        curl = u.diff(x) + v.diff(y)

        # set equations
        self.equations = {}
        
        self.equations["continuity"] = (
            1/rho_r_norm * ( (rho * u).diff(x) +
                             (rho * v).diff(y) )
            + u.diff(x) 
            + v.diff(y)
        )

        self.equations["momentum_x"] = (
            ( a_0**2 / (R * T_r) ) * (
                2 * rho * u * u.diff(x) +
                u**2 * rho.diff(x) + 
                rho * u * v.diff(y) + 
                rho * v * u.diff(y) + 
                u * v * rho.diff(y)
            ) +
            ( (a_0**2 * rho_r_norm) / (R * T_r) ) * (
                2 * u * u.diff(x) + 
                u * v.diff(y) + 
                v * u.diff(y)
            ) + 
            ( 1 / T_r_norm ) * (
                T * rho.diff(x) + 
                rho * T.diff(x)
            ) +
            rho.diff(x) + 
            (rho_r_norm/T_r_norm) * rho.diff(x) - 
            ( (a_0* nu) /(R * T_r * L) ) * (
                2   * (rho * u.diff(x)).diff(x) + 
                      (rho * u.diff(y)).diff(y) +
                      (rho * v.diff(x)).diff(y) -
                2/3 * (rho * curl).diff(x)
            ) -
            ( (a_0* rho_r_norm * nu) /(R * T_r * L) ) * (
                2   * (u.diff(x)).diff(x) + 
                      (u.diff(y)).diff(y) +
                      (v.diff(x)).diff(y) -
                2/3 * (rho * curl).diff(x)
            )
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
            (rho_r_norm/T_r_norm) * rho.diff(y)  - 
            ( (a_0* nu) /(R * T_r * L) ) * (
                2   * (rho * v.diff(y)).diff(y) + 
                      (rho * v.diff(x)).diff(x) +
                      (rho * u.diff(y)).diff(x) -
                2/3 * (rho * curl).diff(y)
            ) -
            ( (a_0* rho_r_norm * nu) /(R * T_r * L) ) * (
                2   * (v.diff(y)).diff(y) + 
                      (v.diff(x)).diff(x) +
                      (u.diff(y)).diff(x) -
                2/3 * (rho * curl).diff(y)
            )
        )
        
        rev_curl = u.diff(y) + v.diff(x)
            
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
             ) -
            ( (a_0 * nu)/(L * rho_r_norm * T_r * c_v) ) * (
                (u*rho * rev_curl).diff(y) +
                (v*rho * rev_curl).diff(x) + 
                2 * (v * rho * v.diff(y)).diff(y) +
                2 * (u * rho * u.diff(x)).diff(x) -
                2/3 * (u * rho * rev_curl).diff(x) -
                2/3 * (v * rho * rev_curl).diff(y)
            ) -
            ( (a_0 * nu)/(L * T_r * c_v) ) * (
                (u * rev_curl).diff(y) +
                (v * rev_curl).diff(x) + 
                2 * (v * v.diff(y)).diff(y) +
                2 * (u * u.diff(x)).diff(x) -
                2/3 * (u * rev_curl).diff(x) -
                2/3 * (v * rev_curl).diff(y)
            ) - 
            ( k / (L* rho_r * T_r_norm * c_v * a_0) ) * (
                (T.diff(x)).diff(x) +
                (T.diff(y)).diff(y)
            )
        )