
# File paths for mesh and boundary data
mesh_files = {
    'MESH_DIRECTORY': 'Meshes/Channel/Coarse/mesh.xdmf',
    'FACET_DIRECTORY': 'Meshes/Channel/Coarse/facet.xdmf'
}

# Specify type of boundaries
boundary_markers = {
    'INFLOW': [4],
    'OUTFLOW': [2],
    'WALLS': [1, 3],
    'SYMMETRY': None
}

# Initial conditions
initial_conditions = {
    'U': (5.0, 0.0), # [m/s]
    'P': 2.0,
    'K': 1.5,
    'E': 2.23,
    'NU_TILDE': 1e-6
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': None,
        'P': 2.0,
        'K': None,
        'E': None,
        'NU_TILDE': None
    },
    'OUTFLOW':{
        'U': None,
        'P': 0.0,
        'K': None,
        'E': None,
        'NU_TILDE': None
    },
    'WALLS':{
        'U': (0.0, 0.0),
        'P': None,
        'K': 0.0,
        'E': None,
        'NU_TILDE': 0.0
    },
    'SYMMETRY':{
        'U': None,
        'P': None,
        'K': None,
        'E': None,
        'NU_TILDE': None
    }
}

# Physical quantities
physical_prm = {
    'VISCOSITY': 0.00181818,
    'FORCE': (0.0, 0.0),
    'STEP_SIZE': 0.005
}

# Simulation parameters
simulation_prm = {
    'QUADRATURE_DEGREE': 2,
    'MAX_ITERATIONS': 5000,
    'TOLERANCE': 1e-6,
    'CFL_RELAXATION': 0.10
}

# Specify where results are saved for k-epsilon runs
saving_directory = {
    'PVD_FILES': 'Results/Channel_k-epsilon/PVD files/',
    'H5_FILES':  'Results/Channel_k-epsilon/H5 files/',
    'RESIDUALS': 'Results/Channel_k-epsilon/Residual files/'
}

# Dedicated saving directories for SA runs
saving_directory_SA = {
    'PVD_FILES': 'Results/Channel_SA/PVD files/',
    'H5_FILES':  'Results/Channel_SA/H5 files/',
    'RESIDUALS': 'Results/Channel_SA/Residual files/',
    'FIGURES':   'Results/Channel_SA/Figures/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}