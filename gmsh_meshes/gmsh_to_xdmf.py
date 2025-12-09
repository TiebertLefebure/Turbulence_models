from mpi4py import MPI
from dolfinx.io import gmsh, XDMFFile

comm = MPI.COMM_WORLD

# Read .msh into MeshData
mesh_data = gmsh.read_from_msh("u_bend_2d.msh", comm, 0, gdim=2)

mesh = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

# Write domain mesh + cell tags
with XDMFFile(comm, "mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    if cell_tags is not None:
        xdmf.write_meshtags(cell_tags, mesh.geometry, "/Xdmf/Domain/Grid/Geometry")

# Write same mesh + facet tags in separate file
with XDMFFile(comm, "facet.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    if facet_tags is not None:
        xdmf.write_meshtags(facet_tags, mesh.geometry, "/Xdmf/Domain/Grid/Geometry")

if comm.rank == 0:
    print("Wrote mesh.xdmf and facet.xdmf")



#docker run -ti --rm \
#  -v $(pwd):/root/shared \
#  -w /root/shared \
#  dolfinx/dolfinx:stable


#python3 gmsh_to_xdmf.py

#mesh.xdmf
#mesh.h5
#facet.xdmf
#facet.h5
