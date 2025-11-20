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
            bcu.append(DirichletBC(V, cond_u, marked_facets, marker))
        # Pressure BC
        cond_p = boundary_conditions[boundary_name].get('P')
        if cond_p is not None:
            bcp.append(DirichletBC(Q, cond_p, marked_facets, marker))
        # SA model: reuse 'K' entry for nu_tilde BCs (e.g., 0 at walls)
        cond_nt = boundary_conditions[boundary_name].get('NU_TILDE')
        if cond_nt is not None:
            bcn.append(DirichletBC(N, cond_nt, marked_facets, marker))

# Initialize constants and expressions
nu = Constant(physical_prm['VISCOSITY'])
force = Constant(physical_prm['FORCE'])
dt = Constant(physical_prm['STEP_SIZE'])
height = mesh.coordinates()[:, 1].max() - mesh.coordinates()[:, 1].min()
y = Expression('H/2 - abs(H/2 - x[1])', H=height, degree=2)

# Initialize functions
u, v, u1, u0 = initialize_functions(V, Constant(initial_conditions['U']))
p, q, p1, p0 = initialize_functions(Q, Constant(initial_conditions['P']))

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
    xmin = mesh.coordinates()[:, 0].min(); xmax = mesh.coordinates()[:, 0].max()
    ymin = mesh.coordinates()[:, 1].min(); ymax = mesh.coordinates()[:, 1].max()

    # Wall-normal profiles (along y-direction) at x = x_mid
    x_mid = 0.5 * (xmin + xmax) # probe where u(y)-profile is plotted (streamwise location x)
    ys = np.linspace(ymin, ymax, 200) # amount of samples in y-direction
    ux = []
    nt = []
    valid_y = []
    for yv in ys:
        try:
            uval = u1(Point(x_mid, yv))
            ux.append(uval[0])
            nt.append(turbulence_model.nu_tilde1(Point(x_mid, yv)))
            valid_y.append(yv)
        except:
            pass
    fig1 = plt.figure()
    plt.plot(ux, valid_y)
    plt.gca().invert_yaxis()
    plt.xlabel('u_x')
    plt.ylabel('y')
    plt.title('Wall-normal velocity profile u(y) at x = x_mid')
    fig2 = plt.figure()
    plt.plot(nt, valid_y)
    plt.gca().invert_yaxis()
    plt.xlabel('nu_tilde')
    plt.ylabel('y')
    plt.title('Wall-normal turbulent viscosity profile nu_tilde(y) at x = x_mid')
    os.makedirs(saving_directory_SA['FIGURES'], exist_ok=True)
    fig1.savefig(os.path.join(saving_directory_SA['FIGURES'], 'wallnormal_u(y)_xmid.png'), dpi=200, bbox_inches='tight')
    fig2.savefig(os.path.join(saving_directory_SA['FIGURES'], 'wallnormal_nu_tilde(y)_xmid.png'), dpi=200, bbox_inches='tight')

    # Centerline streamwise profiles (along x-direction) at y = y_mid
    y_mid = 0.5 * (ymin + ymax)
    xs = np.linspace(xmin, xmax, 200) # amount of samples in x-direction
    ux_center = []
    nt_center = []
    valid_x = []
    for xv in xs:
        try:
            uval = u1(Point(xv, y_mid))
            ux_center.append(uval[0])
            nt_center.append(turbulence_model.nu_tilde1(Point(xv, y_mid)))
            valid_x.append(xv)
        except:
            pass
    fig3 = plt.figure()
    plt.plot(valid_x, ux_center)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Centerline streamwise velocity u(x) at y = y_mid')
    # turn off scientific notation and offsets on the y-axis
    # ax = plt.gca()
    # ax.yaxis.get_major_formatter().set_useOffset(False)
    # ax.yaxis.get_major_formatter().set_scientific(False)
    fig4 = plt.figure()
    plt.plot(valid_x, nt_center)
    plt.xlabel('x')
    plt.ylabel('nu_tilde(x)')
    plt.title('Centerline streamwise turbulent viscosity nu_tilde(x) at y = y_mid')
    # turn off scientific notation and offsets on the y-axis
    # ax = plt.gca()
    # ax.yaxis.get_major_formatter().set_useOffset(False)
    # ax.yaxis.get_major_formatter().set_scientific(False)
    fig3.savefig(os.path.join(saving_directory_SA['FIGURES'], 'centerline_u(x)_ymid.png'), dpi=200, bbox_inches='tight')
    fig4.savefig(os.path.join(saving_directory_SA['FIGURES'], 'centerline_nu_tilde(x)_ymid.png'), dpi=200, bbox_inches='tight')
    plt.show()

# Save results and residuals
if post_processing['SAVE']==True:
    for key in ['FIGURES', 'PVD_FILES', 'H5_FILES', 'RESIDUALS']:
        os.makedirs(saving_directory_SA[key], exist_ok=True)
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1)
    c = plot(sqrt(dot(u1, u1)))
    plt.colorbar(c)
    plt.title('u magnitude')
    plt.subplot(1,3,2)
    c = plot(p1)
    plt.colorbar(c)
    plt.title('p')
    plt.subplot(1,3,3)
    c = plot(turbulence_model.nu_tilde1)
    plt.colorbar(c)
    plt.title('nu_tilde')
    fig.savefig(os.path.join(saving_directory_SA['FIGURES'], 'fields.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure()
    keys = list(residuals.keys())
    for k in range(len(keys)):
        plt.plot(range(1, len(residuals[keys[k]])+1), residuals[keys[k]], label=keys[k])
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('error (log scale)')
    plt.legend()
    fig.savefig(os.path.join(saving_directory_SA['FIGURES'], 'convergence.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    for (key, f) in solutions.items():
        save_pvd_file(f, saving_directory_SA['PVD_FILES'] + key + '.pvd')
        save_h5_file( f, saving_directory_SA['H5_FILES']  + key + '.h5')

    for (key, f) in residuals.items():
        save_list(f, saving_directory_SA['RESIDUALS'] + key + '.txt')
    with open(os.path.join(saving_directory_SA['RESIDUALS'], 'mass_flow.txt'), 'w') as f:
        f.write(f'inflow_flux {flux_in:.16e}\n')
        f.write(f'outflow_flux {flux_out:.16e}\n')
        f.write(f'net_flux {net_flux:.16e}\n')
        f.write(f'rel_imbalance {rel_imb:.16e}\n')
        f.write(f'area_in {area_in:.16e}\n')
        f.write(f'area_out {area_out:.16e}\n')

# Write summary results file
os.makedirs(os.path.dirname(results_file_SA), exist_ok=True)
iterations_completed = len(residuals['u'])
final_errors = {key: (values[-1] if values else float('nan')) for key, values in residuals.items()}
with open(results_file_SA, 'w') as f:
    f.write('case channel_SA\n')
    f.write(f'iterations {iterations_completed}\n')
    f.write(f'converged {converged}\n')
    f.write(f'elapsed_time {total_time:.6f}\n')
    f.write(f'final_error_u {final_errors["u"]:.16e}\n')
    f.write(f'final_error_p {final_errors["p"]:.16e}\n')
    f.write(f'final_error_nu_tilde {final_errors["nu_tilde"]:.16e}\n')
    f.write(f'inflow_flux {flux_in:.16e}\n')
    f.write(f'outflow_flux {flux_out:.16e}\n')
    f.write(f'net_flux {net_flux:.16e}\n')
    f.write(f'rel_imbalance {rel_imb:.16e}\n')
    f.write(f'area_in {area_in:.16e}\n')
    f.write(f'area_out {area_out:.16e}\n')
