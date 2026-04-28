from simulations.base import BaseGeophysicalModel
import numpy as np
import pygimli.physics.ert as ert
import pygimli as pg


class ERTModel(BaseGeophysicalModel):

    def forward(self):
        elecs = self.get_surface_sensors()

        scheme = ert.createData(elecs=elecs, schemeName='dd')
        print(f"Scheme points: {scheme.size()}")

        data = ert.simulate(
            self.mesh,
            res=self.mesh_properties,
            scheme=scheme,
            noiseLevel=1,
            noiseAbs=1e-6,
            addInverse=True,
            verbose=True 
        )
        self.data = data

    def cleanup(self):
        data = self.data
        
        print(f"Initial data points: {data.size()}")
        
        rhoa = np.array(data['rhoa'])
        err  = np.array(data['err'])
        k    = np.array(data['k'])

        data.markInvalid(~np.isfinite(rhoa))
        data.markInvalid(rhoa <= 0)
        data.markInvalid(rhoa > 1e6)
        data.markInvalid(err > 0.5)
        data.markInvalid(~np.isfinite(k))
        data.markInvalid(np.abs(k) > 1e4)

        data.removeInvalid()

        print(f"Remaining data points: {data.size()}")

    def invert(self, lam=20, max_iter=10):
        self.manager = ert.ERTManager(self.data)

        inv_mesh = self.manager.createMesh(self.data, 
                                           quality=34, 
                                           maxCellArea=0.1,
                                           paraDepth=10)

        self.result = self.manager.invert(
            self.data,
            mesh=inv_mesh,
            lam=lam,
            maxIter=max_iter,
            verbose=True
        )
        return self.result, self.manager 