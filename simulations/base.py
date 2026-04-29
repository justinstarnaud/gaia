import os
import pygimli as pg
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PatchCollection


class BaseGeophysicalModel(ABC):

    def __init__(self, mesh, mesh_properties: np.ndarray, n_sensors=24):
        self.mesh = mesh
        self.n_sensors = n_sensors
        self.mesh_properties = mesh_properties
        self.data = None
        self.result = None
        self.manager = None

    def get_surface_sensors(self, x_start=-35, x_end=35, y_min_surface=None):
        """
        x_start, x_end   : explicit x limits for sensor placement (m)
        y_min_surface    : minimum y to be considered "surface" (excludes ground under dam)
        """
        all_nodes = np.array([[n.x(), n.y()] for n in self.mesh.nodes()])

        # Get boundary nodes only
        boundary_nodes = []
        for bound in self.mesh.boundaries():
            if bound.outside():
                for n in range(bound.nodeCount()):
                    node = bound.node(n)
                    boundary_nodes.append([node.x(), node.y()])
        boundary_nodes = np.unique(boundary_nodes, axis=0)

        # --- Fallbacks if not specified ---
        mesh_y_min = all_nodes[:, 1].min()
        mesh_x_min = all_nodes[:, 0].min()
        mesh_x_max = all_nodes[:, 0].max()

        if x_start is None:
            x_start = mesh_x_min + 0.5
        if x_end is None:
            x_end = mesh_x_max - 0.5
        if y_min_surface is None:
            y_min_surface = mesh_y_min + 0.5  # old behaviour

        # --- Apply all three bounds explicitly ---
        surface_nodes = boundary_nodes[
            (boundary_nodes[:, 1] > y_min_surface) &
            (boundary_nodes[:, 0] >= x_start) &
            (boundary_nodes[:, 0] <= x_end)
        ]

        surface_nodes = surface_nodes[np.argsort(surface_nodes[:, 0])]
        print(f"Available surface nodes: {len(surface_nodes)}")

        # Pick n_sensors evenly spaced
        x_targets = np.linspace(x_start, x_end, self.n_sensors)
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
        for c in list(ax.collections):
            if isinstance(c, (PathCollection, PatchCollection)):
                c.remove()
        for line in list(ax.lines):
            line.remove()
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