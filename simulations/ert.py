from simulations.base import BaseGeophysicalModel
import numpy as np
import pygimli.physics.ert as ert

class ERTModel(BaseGeophysicalModel):

    def forward(self):
        top_nodes = self.get_surface_sensors()
        
        indices = np.linspace(0, len(top_nodes) - 1, self.n_sensors, dtype=int)
        elec_positions = top_nodes[indices]

        # Create the scheme with explicit electrode positions ---
        scheme = ert.createERTData(
            elecs=elec_positions,  
            schemeName='dd'
        )

        data = ert.simulate(
            self.mesh,
            res=self.mesh_properties,
            scheme=scheme,
            noiseLevel=1,
            noiseAbs=1e-6
        )
        self.data = data

    def cleanup(self):
        print(f"Initial data points: {self.data.size()}")

        data = self.data
        data.markInvalid(data['rhoa'] <= 0)
        data.markInvalid(data['r'] <= 0)
        data.markInvalid(data['rhoa'] > 1e6)
        data.markInvalid(data['err'] > 0.5)

        data.removeInvalid()
        print(f"Remaining data points: {data.size()}")
    
        return self.data

    def invert(self, lam=200, max_iter=20):
        self.manager = ert.ERTManager(self.data)

        # Create inversion mesh from electrode positions
        inv_mesh = self.manager.createMesh(self.data, quality=34, maxCellArea=0.5)

        self.result = self.manager.invert(
            self.data,
            mesh=inv_mesh,
            lam=lam,
            maxIter=max_iter,
            verbose=True
        )
        return self.result, self.manager