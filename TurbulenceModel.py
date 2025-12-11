from dolfin import *
from Utilities import *




################################################
### Lam-Bremhorst k-epsilon turbulence model ###
################################################

class KEpsilonGeneral:
    def __init__(self, K, bck, bce, k_init, e_init, nu, force, custom_dx, custom_ds, distance_field):
        # K-epsilon model parameters
        self._K = K
        self._bck = bck
        self._bce = bce
        self._k_init = k_init
        self._e_init = e_init

        # General parameters
        self._nu = nu
        self._force = force
        self._dx = custom_dx
        self._ds = custom_ds
        self._y = distance_field  # distance function

        # Initialize all functions
        self._construct_functions()

    def _construct_functions(self):
        """Construct model functions."""
        self._k, self._phi, self._k1, self._k0 = initialize_functions(self._K, Constant(self._k_init))
        self._e, self._psi, self._e1, self._e0 = initialize_functions(self._K, Constant(self._e_init))

    def construct_forms(self):
        raise NotImplementedError("This method must be implemented in subclasses.")
    
    def solve_turbulence_model(self):
        """Solves the turbulence model equations for k and e."""
        # solve k 
        A_K = assemble(self._a_k); b_k = assemble(self._l_k)
        [bc.apply(A_K,b_k) for bc in self._bck]
        solve(A_K, self._k1.vector(), b_k)

        # solve e
        A_E = assemble(self._a_e); b_e = assemble(self._l_e)
        [bc.apply(A_E,b_e) for bc in self._bce]
        solve(A_E, self._e1.vector(), b_e)

        # bound from bellow
        self._k1 = bound_from_bellow(self._k1, 1e-16)
        self._e1 = bound_from_bellow(self._e1, 1e-16)

    def update_variables(self, relaxation = 1.0):
        """Update k and e variables with relaxation."""
        self._k0.assign(relaxation * self._k1 + (1.0 - relaxation) * self._k0)
        self._e0.assign(relaxation * self._e1 + (1.0 - relaxation) * self._e0)

    def _construct_turbulent_quantities(self, external_u1):
        """Construct turbulent quantities used in variational forms."""
        def Max(a, b): return (a+b+abs(a-b))/Constant(2)
        def Min(a, b): return (a+b-abs(a-b))/Constant(2)

        Re_t = (1. / self._nu) * (self._k0**2 / self._e0) 
        Re_k = (1. / self._nu) * (sqrt(self._k0) * self._y) 

        f_nu = Max(Min((1 - exp(- 0.0165 * Re_k))**2 * (1 + 20.5 / Re_t), Constant(1.0)), Constant(0.01116225))
        f_1  = 1 + (0.05 / f_nu)**3
        f_2  = Max(Min(1 - exp(- Re_t**2), Constant(1.0)), Constant(0.0))
        S_sq = 2 * inner(sym(nabla_grad(external_u1)), sym(nabla_grad(external_u1)))

        # Define turbulent quantities
        self._nu_t = 0.09 * f_nu * (self._k0**2 / self._e0)
        self._prod_k = self._nu_t * S_sq
        self._react_k = self._e0 / self._k0
        self._prod_e = 1.44 * self._react_k * f_1 * self._prod_k
        self._react_e = 1.92 * self._react_k * f_2

    # Properties which are refered to in main
    @property
    def nu_t(self):
        return self._nu_t
    
    @property
    def k0(self):
        return self._k0
    
    @property
    def k1(self):
        return self._k1
    
    @property
    def e0(self):
        return self._e0
    
    @property
    def e1(self):
        return self._e1


class KEpsilonSteadyState(KEpsilonGeneral):
    def __init__(self, K, bck, bce, k_init, e_init, nu, force, custom_dx, custom_ds, distance_field):
        super().__init__(K, bck, bce, k_init, e_init, nu, force, custom_dx, custom_ds, distance_field)   
                
    def construct_forms(self, external_u1):
        self._construct_turbulent_quantities(external_u1)

        # Transport equation for k
        FK  = dot(dot(external_u1, nabla_grad(self._k)), self._phi)*self._dx \
            + inner((self._nu + self._nu_t / 1.0) * grad(self._k), grad(self._phi))*self._dx \
            - dot(self._prod_k, self._phi)*self._dx \
            + dot(self._react_k * self._k, self._phi)*self._dx
        self._a_k = lhs(FK); self._l_k = rhs(FK)

        # Transport equation for e        
        FE  = dot(dot(external_u1, nabla_grad(self._e)), self._psi)*self._dx \
            + inner((self._nu + self._nu_t / 1.3) * grad(self._e), grad(self._psi))*self._dx \
            - dot(self._prod_e, self._psi)*self._dx \
            + dot(self._react_e * self._e, self._psi)*self._dx
        self._a_e = lhs(FE); self._l_e = rhs(FE)


class KEpsilonTransient(KEpsilonGeneral):
    def __init__(self, K, bck, bce, k_init, e_init, nu, force, custom_dx, custom_ds, dt, distance_field):
        self._dt = dt
        super().__init__(K, bck, bce, k_init, e_init, nu, force, custom_dx, custom_ds, distance_field)       
        
    def construct_forms(self, external_u1):
        self._construct_turbulent_quantities(external_u1)

        # Transport equation for k
        FK  = dot((self._k - self._k0) / self._dt, self._phi)*self._dx \
            + dot(dot(external_u1, nabla_grad(self._k)), self._phi)*self._dx \
            + inner((self._nu + self._nu_t / 1.0) * grad(self._k), grad(self._phi))*self._dx \
            - dot(self._prod_k, self._phi)*self._dx \
            + dot(self._react_k * self._k, self._phi)*self._dx
        self._a_k = lhs(FK); self._l_k = rhs(FK)

        # Transport equation for e        
        FE  = dot((self._e - self._e0) / self._dt, self._psi)*self._dx \
            + dot(dot(external_u1, nabla_grad(self._e)), self._psi)*self._dx \
            + inner((self._nu + self._nu_t / 1.3) * grad(self._e), grad(self._psi))*self._dx \
            - dot(self._prod_e, self._psi)*self._dx \
            + dot(self._react_e * self._e, self._psi)*self._dx
        self._a_e = lhs(FE); self._l_e = rhs(FE)







#########################################
### Spalart-Allmaras turbulence model ###
#########################################

class SpalartAllmarasGeneral:
    def __init__(self, N, bcn, nu_tilde_init, nu, force, custom_dx, custom_ds, distance_field):
        """Base class for the Spalart-Allmaras one-equation turbulence model."""
        self._N = N
        self._bcn = bcn
        self._nu_tilde_init = nu_tilde_init

        self._nu = nu
        self._force = force
        self._dx = custom_dx
        self._ds = custom_ds
        self._y = distance_field

        self._construct_functions()

    def _construct_functions(self):
        """Construct model functions for the modified turbulent viscosity."""
        self._nu_tilde, self._xi, self._nu_tilde1, self._nu_tilde0 = initialize_functions(self._N, Constant(self._nu_tilde_init))

    def construct_forms(self):
        """Constructs the variational forms. Must be implemented in subclasses."""
        raise NotImplementedError("This method must be implemented in subclasses.")

    def solve_turbulence_model(self):
        """Solves the transport equation for nu_tilde."""
        A_NT = assemble(self._a_nt); b_nt = assemble(self._l_nt)
        [bc.apply(A_NT,b_nt) for bc in self._bcn]
        solve(A_NT, self._nu_tilde1.vector(), b_nt)

        # Enforce positivity
        self._nu_tilde1 = bound_from_bellow(self._nu_tilde1, 1e-16)

    def update_variables(self, relaxation = 1.0):
        """Update nu_tilde variable with relaxation."""
        # The assign method can handle linear combinations of Functions.
        # This is more readable and consistent with the k-epsilon implementation.
        self._nu_tilde0.assign(relaxation * self._nu_tilde1 + (1.0 - relaxation) * self._nu_tilde0)

    def _construct_turbulent_quantities(self, external_u1):
        """
        Constructs the various terms (production, destruction, etc.) for the
        Spalart-Allmaras transport equation.
        This uses a stable formulation where production is an explicit source
        and destruction is an implicit sink.
        """
        # Helper for min function in UFL
        def Min(a, b): return (a+b-abs(a-b))/Constant(2)

        # Non-dimensional viscosity ratio
        chi = self._nu_tilde0 / self._nu
        
        # Damping functions
        f_v1 = chi**3 / (chi**3 + Constant(7.1)**3)
        f_v2 = 1 - chi / (1 + chi * f_v1)
        f_t2 = 1.2 * exp(-0.5 * chi**2)

        # Strain rate magnitude
        S_sq = 2 * inner(sym(nabla_grad(external_u1)), sym(nabla_grad(external_u1)))
        S = sqrt(S_sq + DOLFIN_EPS) # Add epsilon for robustness

        # Wall distance with safety epsilon
        y_safe = self._y + DOLFIN_EPS
        kappa = 0.41

        # Modified strain rate S_tilde
        S_tilde = S + self._nu_tilde0 / (kappa**2 * y_safe**2) * f_v2

        # Argument for f_w function
        r_arg = self._nu_tilde0 / (S_tilde * kappa**2 * y_safe**2 + DOLFIN_EPS)
        r = Min(r_arg, Constant(10.0)) # Cap r as in original model to prevent singularity

        # Wall function f_w
        g = r + 0.3 * (r**6 - r) # Note: c_w2 = 0.3
        cw3 = 2.0
        f_w = g * ((1 + cw3**6) / (g**6 + cw3**6))**(1.0/6.0)

        # Turbulent eddy viscosity for the RANS equations
        self._nu_t = self._nu_tilde0 * f_v1

        # --- Terms for the nu_tilde transport equation ---
        # Model constants
        sigma = 2.0/3.0
        cb1 = 0.1355
        cb2 = 0.622
        cw1 = cb1/kappa**2 + (1 + cb2)/sigma

        # Production term (explicit source)
        # P = cb1 * (1 - f_t2) * S_tilde * nu_tilde
        prod_nt = cb1 * (1 - f_t2) * S_tilde * self._nu_tilde0
        
        # Destruction term (linearized for implicit sink)
        # D = cw1 * f_w * (nu_tilde/y)^2 ~= (cw1 * f_w * nu_tilde_0 / y^2) * nu_tilde
        self._react_nt = cw1 * f_w * (self._nu_tilde0 / y_safe**2)
        
        # Cross-diffusion term (explicit source)
        cross_diff_nt = (cb2/sigma) * inner(nabla_grad(self._nu_tilde0), nabla_grad(self._nu_tilde0))

        # Combine all explicit source terms
        self._source_nt = prod_nt + cross_diff_nt

    @property
    def nu_t(self):
        """Turbulent eddy viscosity."""
        return self._nu_t

    @property
    def nu_tilde0(self):
        """Value of nu_tilde from previous iteration."""
        return self._nu_tilde0

    @property
    def nu_tilde1(self):
        """Value of nu_tilde for current iteration."""
        return self._nu_tilde1


class SpalartAllmarasSteadyState(SpalartAllmarasGeneral):
    def __init__(self, N, bcn, nu_tilde_init, nu, force, custom_dx, custom_ds, distance_field):
        super().__init__(N, bcn, nu_tilde_init, nu, force, custom_dx, custom_ds, distance_field)

    def construct_forms(self, external_u1):
        self._construct_turbulent_quantities(external_u1)

        sigma = 2.0/3.0

        # Weak form for steady-state SA model
        FNT  = dot(dot(external_u1, nabla_grad(self._nu_tilde)), self._xi)*self._dx \
            + inner((self._nu + self._nu_tilde0) / sigma * grad(self._nu_tilde), grad(self._xi))*self._dx \
            + dot(self._react_nt * self._nu_tilde, self._xi)*self._dx \
            - dot(self._source_nt, self._xi)*self._dx
        self._a_nt = lhs(FNT); self._l_nt = rhs(FNT)


class SpalartAllmarasTransient(SpalartAllmarasGeneral):
    def __init__(self, N, bcn, nu_tilde_init, nu, force, custom_dx, custom_ds, dt, distance_field):
        self._dt = dt
        super().__init__(N, bcn, nu_tilde_init, nu, force, custom_dx, custom_ds, distance_field)

    def construct_forms(self, external_u1):
        self._construct_turbulent_quantities(external_u1)

        sigma = 2.0/3.0

        # Weak form for transient SA model
        FNT  = dot((self._nu_tilde - self._nu_tilde0) / self._dt, self._xi)*self._dx \
            + dot(dot(external_u1, nabla_grad(self._nu_tilde)), self._xi)*self._dx \
            + inner((self._nu + self._nu_tilde0) / sigma * grad(self._nu_tilde), grad(self._xi))*self._dx \
            + dot(self._react_nt * self._nu_tilde, self._xi)*self._dx \
            - dot(self._source_nt, self._xi)*self._dx
        self._a_nt = lhs(FNT); self._l_nt = rhs(FNT)