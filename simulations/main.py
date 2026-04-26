from damforge.generate import generate_dataset
from damforge.properties import PropertyArrays
from simulations.base import BaseGeophysicalModel
from simulations.ert import ERTModel
from simulations.seismic import SeismicModel

if __name__ == "__main__":
    scenarios = generate_dataset(n_scenarios=2, output_dir=None, seed=42, write=True)

    for i, (cfg, mesh, labels, per_state) in enumerate(scenarios):
        state: PropertyArrays = per_state[-1]
        models: list[BaseGeophysicalModel] = [
            ERTModel(mesh, state.resistivity_ohm_m, n_sensors=36), 
            SeismicModel(mesh, state.velocity_m_s, n_sensors=48)
            ]

        for model in models:
            model.run()
            model.summary()
            model.save(f"sim_outputs/{model.__class__.__name__}_{i}.png")

