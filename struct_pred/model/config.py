"""
Geometry bin configuration for RNA structure prediction.
Defines distance and angle discretisation used by the prediction heads.
"""

import numpy as np

# Distance discretisation
DIST_MIN = 3.0     # Angstroms
DIST_MAX = 40.0    # Angstroms
DIST_BIN_SIZE = 1.0

# Atom pairs whose pairwise distances are predicted
DISTANCE_ATOMS = ["C3'", "P", "N1", "C4", "C1'", "CiNj", "PiNj"]

# Contact / mask output
CONTACT_ATOMS = ["all atom"]

# Output target specification
GEOMETRY_TARGETS = {
    "2D": {
        "distance": DISTANCE_ATOMS,
        "contact": CONTACT_ATOMS,
    }
}

# Number of bins per geometry type
N_BINS = {
    "2D": {
        "distance": int((DIST_MAX - DIST_MIN) / DIST_BIN_SIZE + 1),   # 38
        "angle": 13,
        "dihedral_angle": 25,
        "contact": 1,
    }
}

# Bin edges for converting distances / angles back to values
_dist_nbins = int((DIST_MAX - DIST_MIN) / DIST_BIN_SIZE + 1)
BINS = {
    "distance": np.linspace(DIST_MIN, DIST_MAX, _dist_nbins),
    "angle": np.linspace(0.0, np.pi, 13),
    "dihedral": np.linspace(-np.pi, np.pi, 25),
}
