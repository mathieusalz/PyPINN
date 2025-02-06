import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import pandas as pd

from modulus.sym.utils.io import InferencerPlotter

# define custom class
class CVP_Comp_Val(InferencerPlotter):
    
    def __init__(self, plot):
        super().__init__()
        
        self.plot = plot
        self.CFD = plot.validator
        self.x_min = plot.x_min
        self.x_max = plot.x_max
        self.y_min = plot.y_min
        self.y_max = plot.y_max

    def __call__(self, invar, pred_outvar):
        "Custom plotting function for validator"
        
        turning_angles = np.unique(self.CFD.turning_angle)
        x_CFD, y_CFD, angle_CFD = self.CFD.x.reshape(1,-1)[0], self.CFD.y.reshape(1,-1)[0], self.CFD.turning_angle.reshape(1,-1)[0]
        N = len(x_CFD)
        
        x_MOD, y_MOD, angle_MOD = pred_outvar["x"][:,0], pred_outvar["y"][:,0], pred_outvar["turning_angle"][:,0]
        
        extent_MOD = (self.x_min, self.x_max, 
                      self.y_min, self.y_max)
                
        quants_to_plot = ['u', 'v', 'rho', 'T']
        
        quantities = []
        
        for key in pred_outvar:
            
            if key in quants_to_plot:
                if key == "u":
                    cfd_quant = self.CFD.u.reshape(1,-1)[0]
                    title = "u"
                elif key == "v":
                    cfd_quant = self.CFD.v.reshape(1,-1)[0]
                    title = "v"
                elif key == "rho":
                    cfd_quant = self.CFD.rho.reshape(1,-1)[0]
                    title = r"\rho"
                elif key == "T":
                    cfd_quant = self.CFD.T.reshape(1,-1)[0]
                    title = "T"

                quantities.append((cfd_quant, pred_outvar[key][:,0], title))

        # Initialize figures and axes for each quantity
        figs = {}
        axes_dict = {}
        for _, _, title in quantities:
            fig, axes = plt.subplots(len(turning_angles[::2]), 3, figsize=(15, len(turning_angles[::2]) * 3.25))
            figs[title] = fig
            axes_dict[title] = axes

        for j, turning_angle in enumerate(turning_angles[::2]):
            
            mask_MOD = angle_MOD == turning_angle
            x_MOD_angle, y_MOD_angle = x_MOD[mask_MOD], y_MOD[mask_MOD]
            
            mask_CFD = angle_CFD == turning_angle
            x_CFD_angle, y_CFD_angle = x_CFD[mask_CFD], y_CFD[mask_CFD]

            for i, quantity in enumerate(quantities):
                
                cfd, pred, title = quantity
                pred_angle = pred[mask_MOD]
                cfd_angle = cfd[mask_CFD]

                MOD, X_MOD, Y_MOD = self.interpolate_output(x_MOD_angle, 
                                                            y_MOD_angle,
                                                            pred_angle,
                                                            extent_MOD,
                )

                CFD, X_CFD, Y_CFD = self.interpolate_output(x_CFD_angle, 
                                                            y_CFD_angle,
                                                            cfd_angle,
                                                            extent_MOD,
                )
                
                ax = axes_dict[title][j][0]
                p = ax.pcolor(X_MOD, Y_MOD, MOD)
                ax.set_title(fr'PINN: ${title}$ for turning angle {turning_angle:.3f}')
                cb = figs[title].colorbar(p, ax=ax)
                ax.plot()
                ax.set_aspect(1)
                ax.set_xlim(self.x_min, self.x_max)
                ax.set_ylim(self.y_min, self.y_max)

                ax = axes_dict[title][j][1]
                p = ax.pcolor(X_CFD, Y_CFD, CFD)
                ax.set_title(fr'CFD: ${title}$ for turning angle {turning_angle:.3f}')
                cb = figs[title].colorbar(p, ax=ax)
                ax.plot()
                ax.set_aspect(1)
                ax.set_xlim(self.x_min, self.x_max)
                ax.set_ylim(self.y_min, self.y_max)

                ax = axes_dict[title][j][2]
                p = ax.pcolor(X_CFD, Y_CFD, abs(CFD-MOD), cmap='hot')
                ax.set_title(fr'Error: ${title}$ for turning angle {turning_angle:.3f}')
                cb = figs[title].colorbar(p, ax=ax)
                ax.plot()
                ax.set_aspect(1)
                ax.set_xlim(self.x_min, self.x_max)
                ax.set_ylim(self.y_min, self.y_max)
                
        return [(fig, title.replace("\\", "")) for title, fig in figs.items()]


    @staticmethod
    def interpolate_output(x, y, us, extent):
        "Interpolates irregular points onto a mesh"

        # define mesh to interpolate onto
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], 100),
            np.linspace(extent[2], extent[3], 100),
            indexing="ij",
        )

        # linearly interpolate points onto mesh
        us = scipy.interpolate.griddata(
            (x, y), us, tuple(xyi)
            )

        return us, xyi[0], xyi[1]
