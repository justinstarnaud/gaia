from damforge.generate import generate_dataset
import pygimli.physics.ert as ert
import numpy as np

def forward(mesh, resistivity, n_electrodes = 24):
    # Get all node coordinates directly from the mesh ---
    all_nodes = np.array([[n.x(), n.y()] for n in mesh.nodes()])

    # Filter for top-surface nodes (highest Y = surface of dam) ---
    y_max = all_nodes[:, 1].max()
    tol = (y_max - all_nodes[:, 1].min()) * 0.01  # 1% of total height
    top_nodes = all_nodes[np.abs(all_nodes[:, 1] - y_max) < tol]
    top_nodes = top_nodes[np.argsort(top_nodes[:, 0])]  # sort left → right

    # Create Elecs
    
    indices = np.linspace(0, len(top_nodes) - 1, n_electrodes, dtype=int)
    elec_positions = top_nodes[indices]

    # Create the scheme with explicit electrode positions ---
    scheme = ert.createERTData(
        elecs=elec_positions,  
        schemeName='dd'
    )

    data = ert.simulate(
        mesh,
        res=resistivity,
        scheme=scheme,
        noiseLevel=1,
        noiseAbs=1e-6
    )

    return data

def cleanup(data):
    # Remove negative or zero apparent resistivity
    data.markInvalid(data['rhoa'] <= 0)
    
    # Remove physically unreasonable values
    # (adjust upper bound based on your expected resistivity range)
    data.markInvalid(data['rhoa'] > 1e6)
    
    # Remove data with very high error estimates
    data.markInvalid(data['err'] > 0.5)  # 50% error threshold
    
    # Drop all marked entries
    data.removeInvalid()
    
    print(f"Remaining data points: {data.size()}")
    
    return data

def invert(data):
    manager = ert.ERTManager(data)

    # Create inversion mesh from electrode positions
    inv_mesh = manager.createMesh(data, quality=34, maxCellArea=0.5)

    result = manager.invert(
        data,
        mesh=inv_mesh,
        lam=20,
        maxIter=20,
        verbose=True
    )

    print(f"Inversion RMS: {manager.inv.chi2():.3f}")

    return result, manager

scenarios = generate_dataset(n_scenarios=1, output_dir=None, seed=42)

ert_inversions = []
for (cfg, mesh, labels, per_state) in scenarios:
    resistivity = per_state[-1].resistivity_ohm_m # Lets say we take the 3rd state which is full of water
    fwd = forward(mesh, resistivity)
    fwd_clean = cleanup(fwd)
    result, manager = invert(fwd_clean)
    ert_inversions.append((result, manager))
    manager.showResult()