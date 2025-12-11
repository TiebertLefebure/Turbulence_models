
# File paths for mesh and boundary data
mesh_files = {
    'MESH_DIRECTORY': 'Meshes/PipeBend/Coarse/mesh.xdmf',
    'FACET_DIRECTORY': 'Meshes/PipeBend/Coarse/facet.xdmf'
}

# Specify type of boundaries
boundary_markers = {
    'INFLOW': [2],
    'OUTFLOW': [3],
    'WALLS': [4]
}

# 2D U-bend (Ansys Manual VMFL048)
# radius = 14 mm
# radius of curvature = 125 mm
# straight section lenghts = 10*D = 280 mm


# inlet bulk velocity U = 1.42 m/s

# choose turbulence intensity I = 5%

# set turbulent length scale l ≈ 0.07*D = 2.0 mm

# initial & inflow boundary conditions for K & E:
# K = 1.5 * (U * I)^2 = 0.008 [m^2/s^2]
# E = Cµ^(3/4) * K^(3/2) / l = 0.054 [m^2/s^3], using Cµ = 0.09



# Initial conditions
initial_conditions = {
    'U': (0.5, 0.0, 0.0), 
    'P': 2.0,
    'K': 0.008,    
    'E': 0.054,    
    'NU_TILDE': 0.0
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': (1.42, 0.0, 0.0), #0.0
        'P': None,  #2.0
        'K': 0.008, #None
        'E': 0.054,  #None
        'NU_TILDE': 2.67e-6 # 3*ν = 2.67e-6
    },
    'OUTFLOW':{
        'U': None,
        'P': 0.0,
        'K': None,
        'E': None,
        'NU_TILDE': None
    },
    'WALLS':{
        'U': (0.0, 0.0, 0.0),
        'P': None,
        'K': 0.0,
        'E': 0.0,
        'NU_TILDE': 0.0
    }
}

# Physical quantities
physical_prm = {
    'VISCOSITY': 8.9e-7,  #0.00181818
    'FORCE': (0.0, 0.0, 0.0),
    'STEP_SIZE': 5e-4 #0.005
}

# Reynolds number:
# Re = U * D / ν = 1.42 * 0.028 / 1.5e-5 ≈ 4.5 × 10^4

# Simulation parameters
simulation_prm = {
    'QUADRATURE_DEGREE': 2,
    'MAX_ITERATIONS': 3000,
    'TOLERANCE': 1e-6, #1e-6
    'CFL_RELAXATION': 0.1 #0.25
}

# Specify where results are saved
saving_directory = {
    'PVD_FILES': 'Results/PipeBend/PVD files/',
    'H5_FILES':  'Results/PipeBend/H5 files/',
    'RESIDUALS': 'Results/PipeBend/Residual files/'
}

# Specify where results are saved for SA model
saving_directory_SA = {
    'PVD_FILES': 'Results/PipeBend_SA/PVD files/',
    'H5_FILES':  'Results/PipeBend_SA/H5 files/',
    'RESIDUALS': 'Results/PipeBend_SA/Residual files/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}




#docker run -ti \
#    -v $(pwd):/home/fenics/shared \
#    -w /home/fenics/shared \
#    quay.io/fenicsproject/stable:current


#cd ~/shared

#ls 

#python3 PipeBendSimulation.py