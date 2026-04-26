from simulations.base import BaseGeophysicalModel
import pygimli.physics.traveltime as tt
import numpy as np

class SeismicModel(BaseGeophysicalModel):

    def forward(self, shot_distance=3):
        top_nodes = self.get_surface_sensors()

        # Pick evenly spaced sensor positions
        indices = np.linspace(0, len(top_nodes) - 1, self.n_sensors, dtype=int)
        sensor_positions = top_nodes[indices]

        # Create refraction scheme
        # shotDistance: every N-th sensor fires as a shot source
        scheme = tt.createRAData(sensor_positions, shotDistance=shot_distance)

        # Simulate — seismic uses slowness (1/velocity), not velocity directly
        self.data = tt.simulate(
            mesh=self.mesh,
            scheme=scheme,
            slowness=1.0 / self.mesh_properties,   # velocity in m/s → slowness in s/m
            secNodes=2,                 # secondary nodes for accuracy
            noiseLevel=0.01,            # 1% relative noise
            noiseAbs=1e-5               # absolute noise floor (s)
        )

        return self.data


    def cleanup(self):
        print(f"Initial data points: {self.data.size()}")

        self.data.markInvalid(self.data['t'] <= 0)
        self.data.markInvalid(self.data['t'] > 1.0)    # > 1s is unrealistic for a dam
        self.data.markInvalid(self.data['err'] > 0.5)
        self.data.removeInvalid()
        print(f"Remaining data points: {self.data.size()}")
        return self.data

    def invert(self, v_top=500, v_bottom=3000, lam=10):
        self.manager = tt.TravelTimeManager(self.data)

        self.result = self.manager.invert(
            self.data,
            secNodes=2,             # secondary nodes, same as forward
            paraMaxCellSize=1.0,    # inversion mesh cell size (tune for granularity)
            maxIter=12,
            lam=lam,
            # gradient starting model: velocity increases with depth
            useGradient=True,
            vTop=v_top,             # expected velocity at surface (m/s)
            vBottom=v_bottom,       # expected velocity at depth (m/s)
            verbose=True
        )

        return self.result, self.manager