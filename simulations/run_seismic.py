from damforge.generate import generate_dataset

import pygimli.physics.traveltime as tt
import numpy as np

def forward_seismic(mesh, velocity, n_sensors=24, shot_distance=3):
    # Extract surface nodes (same approach as ERT)
    all_nodes = np.array([[n.x(), n.y()] for n in mesh.nodes()])
    y_max = all_nodes[:, 1].max()
    tol = (y_max - all_nodes[:, 1].min()) * 0.01
    top_nodes = all_nodes[np.abs(all_nodes[:, 1] - y_max) < tol]
    top_nodes = top_nodes[np.argsort(top_nodes[:, 0])]

    # Pick evenly spaced sensor positions
    indices = np.linspace(0, len(top_nodes) - 1, n_sensors, dtype=int)
    sensor_positions = top_nodes[indices]

    # Create refraction scheme
    # shotDistance: every N-th sensor fires as a shot source
    scheme = tt.createRAData(sensor_positions, shotDistance=shot_distance)

    # Simulate — seismic uses slowness (1/velocity), not velocity directly
    data = tt.simulate(
        mesh=mesh,
        scheme=scheme,
        slowness=1.0 / velocity,   # velocity in m/s → slowness in s/m
        secNodes=2,                 # secondary nodes for accuracy
        noiseLevel=0.01,            # 1% relative noise
        noiseAbs=1e-5               # absolute noise floor (s)
    )

    return data


def cleanup_seismic(data):
    # Remove zero or negative traveltimes
    data.markInvalid(data['t'] <= 0)

    # Remove physically unreasonable traveltimes
    # (adjust based on your profile length and expected velocity range)
    data.markInvalid(data['t'] > 1.0)    # > 1s is unrealistic for a dam

    # Remove high error
    data.markInvalid(data['err'] > 0.5)

    data.removeInvalid()

    print(f"Remaining data points: {data.size()}")

    return data


def invert_seismic(data, v_top=500, v_bottom=3000):
    manager = tt.TravelTimeManager(data)

    result = manager.invert(
        data,
        secNodes=2,             # secondary nodes, same as forward
        paraMaxCellSize=1.0,    # inversion mesh cell size (tune for granularity)
        maxIter=20,
        lam=10,
        # gradient starting model: velocity increases with depth
        useGradient=True,
        vTop=v_top,             # expected velocity at surface (m/s)
        vBottom=v_bottom,       # expected velocity at depth (m/s)
        verbose=True
    )

    print(f"Inversion chi²: {manager.inv.chi2():.3f}")  # ideally ~1.0

    return result, manager


scenarios = generate_dataset(n_scenarios=1, output_dir=None, seed=42)

seismic_data = []
for (cfg, mesh, labels, per_state) in scenarios:
    velocity = per_state[-1].velocity_m_s # Lets say we take the 3rd state which is full of water
    fwd   = forward_seismic(mesh, velocity)
    fwd_clean   = cleanup_seismic(fwd)
    result, manager = invert_seismic(fwd)
    manager.showResult()   # plots velocity section
    print(manager.inv.chi2())   