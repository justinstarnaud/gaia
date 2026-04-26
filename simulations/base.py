import os
import pygimli as pg
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class BaseGeophysicalModel(ABC):

    def __init__(self, mesh, mesh_properties: np.ndarray, n_sensors=24):
        self.mesh = mesh
        self.n_sensors = n_sensors
        self.mesh_properties = mesh_properties
        self.data = None
        self.result = None
        self.manager = None

    # ── Sensor extraction (shared by all surface methods) ─────────────────────
    def get_surface_sensors(self):
        all_nodes = np.array([[n.x(), n.y()] for n in self.mesh.nodes()])
        y_max = all_nodes[:, 1].max()
        tol = 0.1  # fixed small tolerance in mesh units, not relative
        
        # Get true crest nodes only
        top_nodes = all_nodes[np.abs(all_nodes[:, 1] - y_max) < tol]
        top_nodes = top_nodes[np.argsort(top_nodes[:, 0])]
        
        # Uniform spatial spacing, not index spacing
        x_min, x_max = top_nodes[:, 0].min(), top_nodes[:, 0].max()
        x_targets = np.linspace(x_min, x_max, self.n_sensors)
        
        # Snap each target x to nearest actual node
        sensors = []
        for xt in x_targets:
            idx = np.argmin(np.abs(top_nodes[:, 0] - xt))
            sensors.append(top_nodes[idx])
        
        return np.array(sensors)

    @abstractmethod
    def forward(self, model):
        """Run forward simulation. model = resistivity (ERT) or velocity (seismic)."""
        pass

    @abstractmethod
    def cleanup(self):
        """Remove invalid data points."""
        pass

    @abstractmethod
    def invert(self, **kwargs):
        """Run inversion on self.data."""
        pass

    def save(self, path: str):
        path = os.path.abspath(path)
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        plt.switch_backend('Agg')  # disable GUI window
        ax, cb = pg.show(
            self.manager.paraDomain,
            self.result,
            colorBar=True,
            label="Resistivity (Ω·m)",
        )
        fig = ax.get_figure()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved to {path}")

    def run(self, **invert_kwargs):
        print(f"-----Forward Pass for {self.__class__.__name__}-----")
        self.forward()
  
        print(f"-----Cleanup for {self.__class__.__name__}-----")
        self.cleanup()

        print(f"-----Inverse pass for {self.__class__.__name__}-----")
        self.invert(**invert_kwargs)

        return self.result

    def chi2(self):
        if self.manager is None:
            raise RuntimeError("No inversion run yet.")
        return self.manager.inv.chi2()
    
    def summary(self):
        print(f"Method   : {self.__class__.__name__}")
        print(f"Sensors  : {self.n_sensors}")
        print(f"Data pts : {self.data.size() if self.data else 'N/A'}")
        if self.manager:
            print(f"chi²     : {self.chi2():.3f}")