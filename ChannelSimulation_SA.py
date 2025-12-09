from dolfin import *
from Utilities import *
from Configs.ConfigChannel import *
from TurbulenceModel import SpalartAllmarasTransient as SA
import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Load mesh 
[mesh, marked_facets] = load_mesh_from_file(mesh_files['MESH_DIRECTORY'], mesh_files['FACET_DIRECTORY'])

# Custom integration measures
quadrature_degree = simulation_prm['QUADRATURE_DEGREE']
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
ds = Measure("ds", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
ds_m = Measure("ds", domain=mesh, subdomain_data=marked_facets, metadata={"quadrature_degree": quadrature_degree})

# Construct periodic boundary condition
mesh_width  = mesh.coordinates()[:, 0].max() - mesh.coordinates()[:, 0].min()
class Periodic(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)
    def map(self, x, y):
        y[0] = x[0] - mesh_width
        y[1] = x[1]
periodic = Periodic(1E-5)

# Construct function spaces
V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain = periodic)       
Q = FunctionSpace(mesh, "CG", 1)                                        
N = FunctionSpace(mesh, "CG", 1, constrained_domain = periodic)  

# Construct boundary conditions
bcu=[]; bcp=[]; bcn=[]

for boundary_name, markers in boundary_markers.items():
    if markers is None:
        continue  

    for marker in markers:
        # Velocity BC
        cond_u = boundary_conditions[boundary_name].get('U')
        if cond_u is not None:
            bc_u = format_value_for_space(V, cond_u)
            bcu.append(DirichletBC(V, bc_u, marked_facets, marker))
        # Pressure BC
        cond_p = boundary_conditions[boundary_name].get('P')
        if cond_p is not None:
            bc_p = format_value_for_space(Q, cond_p)
            bcp.append(DirichletBC(Q, bc_p, marked_facets, marker))
        # SA model: reuse 'K' entry for nu_tilde BCs (e.g., 0 at walls)
        cond_nt = boundary_conditions[boundary_name].get('NU_TILDE')
        if cond_nt is not None:
            bc_nt = format_value_for_space(N, cond_nt)
            bcn.append(DirichletBC(N, bc_nt, marked_facets, marker))

# Initialize constants and expressions
nu = Constant(physical_prm['VISCOSITY'])
force = format_value_for_space(V, physical_prm['FORCE'])
dt = Constant(physical_prm['STEP_SIZE'])
height = mesh.coordinates()[:, 1].max() - mesh.coordinates()[:, 1].min()
y = Expression('H/2 - abs(H/2 - x[1])', H=height, degree=2)

# Initialize functions
u_init = format_value_for_space(V, initial_conditions['U'])
p_init = format_value_for_space(Q, initial_conditions['P'])
u, v, u1, u0 = initialize_functions(V, u_init)
p, q, p1, p0 = initialize_functions(Q, p_init)

# Initialize turbulence model
turbulence_model = SA(N, bcn, initial_conditions['NU_TILDE'], nu, force, dx, ds, dt, y)
turbulence_model.construct_forms(u1)

# Construct RANS forms
F1  = dot((u - u0) / dt, v)*dx \
    + dot(dot(u0, nabla_grad(u)), v)*dx \
    + inner((nu + turbulence_model.nu_t) * grad(u), grad(v))*dx \
    - dot(force, v)*dx
    
F2  = dot(grad(p), grad(q))*dx + dot(div(u1) / dt, q)*dx
F3  = dot(u, v)*dx - dot(u1, v)*dx + dt * dot(grad(p1), v)*dx

# Precompute lhs and rhs
a_1, l_1 = lhs(F1), rhs(F1)
a_2, l_2 = lhs(F2), rhs(F2)
a_3, l_3 = lhs(F3), rhs(F3)

# Main loop
residuals = {key: [] for key in ['u', 'p', 'nu_tilde']}
start_time = time.time()
converged = False
vel_tolerance = simulation_prm.get('VELOCITY_TOLERANCE', simulation_prm['TOLERANCE'])
default_tolerance = simulation_prm['TOLERANCE']
for iter in range(simulation_prm['MAX_ITERATIONS']):
    # Dynamic time-stepping
    if iter > 0:
        h_x = MaxCellEdgeLength(mesh); h_y = MinCellEdgeLength(mesh)
        step_size = calculate_cfl_time_step(u0, h_x, h_y, simulation_prm['CFL_RELAXATION'], mesh)
        dt.assign(Constant(step_size))
        # Calculate maximum CFL number
        u_x = u0[0]
        u_y = u0[1]
        cfl_expr = step_size * (abs(u_x) / h_x + abs(u_y) / h_y)
        cfl_max = project(cfl_expr, FunctionSpace(mesh, "DG", 0)).vector().max()
        print(f"CFL number: {cfl_max:.2f}")

    # Solve NS
    A_1 = assemble(a_1); b_1 = assemble(l_1)
    [bc.apply(A_1,b_1) for bc in bcu]
    solve(A_1, u1.vector(), b_1, 'mumps')

    A_2 = assemble(a_2); b_2 = assemble(l_2)
    [bc.apply(A_2,b_2) for bc in bcp]
    solve(A_2, p1.vector(), b_2, 'mumps')

    A_3 = assemble(a_3); b_3 = assemble(l_3)
    [bc.apply(A_3,b_3) for bc in bcu]
    solve(A_3, u1.vector(), b_3, 'mumps')

    # Solve turbulence model
    turbulence_model.solve_turbulence_model()
    
    # Apply relaxation for better convergence
    #relaxation = 0.9  # Less aggressive relaxation
    #u1.vector()[:] = relaxation * u1.vector() + (1 - relaxation) * u0.vector()
    #p1.vector()[:] = relaxation * p1.vector() + (1 - relaxation) * p0.vector()
    #turbulence_model.update_variables(relaxation)

    # No relaxation - let solution converge naturally
    # u1, p1, and turbulence_model already have the computed values
   
    errors = [
        compute_l2_error(u1, u0),
        compute_l2_error(p1, p0),
        compute_l2_error(turbulence_model.nu_tilde1, turbulence_model.nu_tilde0)
    ]
    tolerances = [vel_tolerance, default_tolerance, default_tolerance]
    break_flag = all(err <= tol for err, tol in zip(errors, tolerances))
    converged = break_flag

    # Update residuals and print summary
    print(f'iter: {iter+1} ({time.time() - start_time:.2f}s) - L2 errors: '
          f'|u1-u0|= {errors[0]:.2e} (required: {tolerances[0]:.2e}), '
          f'|p1-p0|= {errors[1]:.2e} (required: {tolerances[1]:.2e}), '
          f'|nu_tilde1-nu_tilde0|= {errors[2]:.2e} (required: {tolerances[2]:.2e})')

    for key, error in zip(residuals.keys(), errors):
        residuals[key].append(error)

    # Update variables for next iteration
    u0.assign(u1)
    p0.assign(p1)
    turbulence_model.update_variables()

    # Check for convergence
    if break_flag:
        print(f'Simulation converged in {iter+1} iterations ({time.time() - start_time:.2f} seconds)')
        break
total_time = time.time() - start_time

solutions = {'u': u1, 'p': p1, 'nu_tilde': turbulence_model.nu_tilde1}

# Mass flow check
n = FacetNormal(mesh)
def _sum_over(markers):
    if markers is None:
        return 0.0
    total = 0.0
    for m in markers:
        total += assemble(dot(u1, n) * ds_m(m))
    return total
def _area_over(markers):
    if markers is None:
        return 0.0
    total = 0.0
    for m in markers:
        total += assemble(1.0 * ds_m(m))
    return total
flux_in  = _sum_over(boundary_markers.get('INFLOW'))
flux_out = _sum_over(boundary_markers.get('OUTFLOW'))
area_in  = _area_over(boundary_markers.get('INFLOW'))
area_out = _area_over(boundary_markers.get('OUTFLOW'))
net_flux = flux_in + flux_out
rel_imb  = abs(net_flux) / (abs(flux_in) + abs(flux_out) + DOLFIN_EPS)
print(f'Mass flow check: inflow={flux_in:.6e}, outflow={flux_out:.6e}, net={net_flux:.6e}, rel={rel_imb:.3e}, Ubar_in={flux_in/(area_in + DOLFIN_EPS):.6e}, Ubar_out={flux_out/(area_out + DOLFIN_EPS):.6e}')

# Visualize
if post_processing['PLOT']==True:
    visualize_functions(solutions)
    visualize_convergence(residuals)
    

# Save results and residuals
if post_processing['SAVE']==True:
    for (key, f) in solutions.items():
        save_pvd_file(f, saving_directory_SA['PVD_FILES'] + key + '.pvd')
        save_h5_file( f, saving_directory_SA['H5_FILES']  + key + '.h5')

    for (key, f) in residuals.items():
        save_list(f, saving_directory_SA['RESIDUALS'] + key + '.txt')
   
