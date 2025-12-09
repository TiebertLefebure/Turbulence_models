from mpi4py import MPI
from dolfinx.io import XDMFFile
import numpy as np

comm = MPI.COMM_WORLD

# --- Read mesh + cell tags ---
with XDMFFile(comm, "mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    cell_tags = xdmf.read_meshtags(mesh, name="cell_tags")

# FEniCS needs facet entities before reading facet tags
mesh.topology.create_entities(mesh.topology.dim - 1)

# --- Read facet tags ---
with XDMFFile(comm, "facet.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(mesh, name="facet_tags")

if comm.rank == 0:
    print(mesh)
    print("Unique cell tags:", np.unique(cell_tags.values))
    print("Unique facet tags:", np.unique(facet_tags.values))


#python3 test_mesh.py


#Unique cell tags: [1]
#Unique facet tags: [2 3 4]
