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
        
        # Get boundary nodes only (nodes on the outer edge of the mesh)
        boundary_nodes = []
        for bound in self.mesh.boundaries():
            if bound.outside():   # only exterior boundary edges
                for n in range(bound.nodeCount()):
                    node = bound.node(n)
                    boundary_nodes.append([node.x(), node.y()])
        
        boundary_nodes = np.unique(boundary_nodes, axis=0)
        
        # Exclude the bottom (foundation) — keep only dam surface
        y_min = all_nodes[:, 1].min()
        x_min = all_nodes[:, 0].min()
        x_max = all_nodes[:, 0].max()
        
        surface_nodes = boundary_nodes[
            (boundary_nodes[:, 1] > y_min + 0.5) &  # not the base
            (boundary_nodes[:, 0] > x_min + 0.5) &  # not left wall
            (boundary_nodes[:, 0] < x_max - 0.5)     # not right wall
        ]
        
        surface_nodes = surface_nodes[np.argsort(surface_nodes[:, 0])]
        print(f"Available surface nodes: {len(surface_nodes)}")
        
        # Pick n_sensors evenly spaced from those
        x_targets = np.linspace(surface_nodes[:,0].min(), 
                                surface_nodes[:,0].max(), 
                                self.n_sensors)
        sensors = []
        for xt in x_targets:
            idx = np.argmin(np.abs(surface_nodes[:, 0] - xt))
            sensors.append(surface_nodes[idx])
        
        _, unique_idx = np.unique([s[0] for s in sensors], return_index=True)
        sensors = np.array(sensors)[sorted(unique_idx)]
        print(f"Final sensor count: {len(sensors)}")
        return sensors

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
        ax, cb = self.manager.showResult()
        ax.set_ylim(-25, 0)   # y is negative-downward in pyGIMLi
        ax.figure.tight_layout()
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