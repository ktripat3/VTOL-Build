import sympy as sp
import numpy as np

def get_phase_trajectory_7(H_start, H_end, R_start, Vx_start, Vx_end, t_phase, t_offset):
    """
    Generates altitude H(t) and range R(t) profiles for a phase (climb, cruise, or descent)
    using 7th-order (vertical) and 6th-order (horizontal) polynomials.

    Inputs:
        H_start   : initial altitude [m]
        H_end     : final altitude [m]
        R_start   : initial horizontal position [m]
        Vx_start  : initial horizontal velocity [m/s]
        Vx_end    : final horizontal velocity [m/s]
        t_phase   : phase duration [s]
        t_offset  : phase shift (used in descent profiles)

    Outputs:
        H_sym  : SymPy expression for altitude [m]
        R_sym  : SymPy expression for range [m]
        Vx_sym : SymPy horizontal velocity [m/s]
        Vy_sym : SymPy vertical velocity [m/s]
        Ax_sym : SymPy horizontal acceleration [m/s²]
        Ay_sym : SymPy vertical acceleration [m/s²]
        t      : SymPy time symbol
    """
    t = sp.symbols('t')
    t1 = t_phase

    # -----------------------------
    # Vertical Trajectory
    # -----------------------------
    H_powers = [t**i for i in range(8)]
    y_coeffs = sp.symbols('y0:8')
    H_poly = sum(c*p for c,p in zip(y_coeffs,H_powers))

    Vy_poly = sp.diff(H_poly,t)
    Ay_poly = sp.diff(Vy_poly,t)
    Jy_poly = sp.diff(Ay_poly,t)

    H_eqs = [
        H_poly.subs(t,0) - H_start,
        H_poly.subs(t,t1) - H_end,
        Vy_poly.subs(t,0) - 0,
        Vy_poly.subs(t,t1) - 0,
        Ay_poly.subs(t,0) - 0,
        Ay_poly.subs(t,t1) - 0,
        Jy_poly.subs(t,0) - 0,
        Jy_poly.subs(t,t1) - 0
    ]

    H_mat = sp.linear_eq_to_matrix(H_eqs, y_coeffs)
    y_sol = sp.solve_linear_system(H_mat[0].row_join(sp.Matrix(H_mat[1])), *y_coeffs)
    H_poly = H_poly.subs(y_sol)
    H_sym = sp.simplify(H_poly).subs(t, t - t_offset)
    Vy_sym = sp.diff(H_sym, t)
    Ay_sym = sp.diff(Vy_sym, t)

    # -----------------------------
    # Horizontal Trajectory
    # -----------------------------
    R_powers = [t**i for i in range(7)]
    x_coeffs = sp.symbols('x0:7')
    R_poly = sum(c*p for c,p in zip(x_coeffs,R_powers))

    Vx_poly = sp.diff(R_poly,t)
    Ax_poly = sp.diff(Vx_poly,t)
    Jx_poly = sp.diff(Ax_poly,t)

    R_eqs = [
        R_poly.subs(t,0) - R_start,
        Vx_poly.subs(t,0) - Vx_start,
        Vx_poly.subs(t,t1) - Vx_end,
        Ax_poly.subs(t,0) - 0,
        Ax_poly.subs(t,t1) - 0,
        Jx_poly.subs(t,0) - 0,
        Jx_poly.subs(t,t1) - 0
    ]

    R_mat = sp.linear_eq_to_matrix(R_eqs, x_coeffs)
    x_sol = sp.solve_linear_system(R_mat[0].row_join(sp.Matrix(R_mat[1])), *x_coeffs)
    R_poly = R_poly.subs(x_sol)
    R_sym = sp.simplify(R_poly).subs(t, t - t_offset)
    Vx_sym = sp.diff(R_sym, t)
    Ax_sym = sp.diff(Vx_sym, t)

    return H_sym, R_sym, Vx_sym, Vy_sym, Ax_sym, Ay_sym, t

def get_phase_trajectory_11(H_start, H_end, R_start, Vx_start, Vx_end, t_phase, t_offset):
    """
    Generates altitude H(t) and range R(t) profiles using
    an 11th-order vertical polynomial and 6th-order horizontal polynomial.
    """
    t = sp.symbols('t')
    t1 = t_phase


    # -----------------------------
    # Horizontal Trajectory (6th order)
    # -----------------------------
    R_powers = [t**i for i in range(7)]
    x_coeffs = sp.symbols('x0:7')
    R_poly = sum(c*p for c,p in zip(x_coeffs,R_powers))

    Vx_poly = sp.diff(R_poly,t)
    Ax_poly = sp.diff(Vx_poly,t)
    Jx_poly = sp.diff(Ax_poly,t)

    R_eqs = [
        R_poly.subs(t,0) - R_start,
        Vx_poly.subs(t,0) - Vx_start,
        Vx_poly.subs(t,t1) - Vx_end,
        Ax_poly.subs(t,0),
        Ax_poly.subs(t,t1),
        Jx_poly.subs(t,0),
        Jx_poly.subs(t,t1)
    ]

    R_mat = sp.linear_eq_to_matrix(R_eqs, x_coeffs)
    x_sol = sp.solve_linear_system(R_mat[0].row_join(sp.Matrix(R_mat[1])), *x_coeffs)

    R_poly = R_poly.subs(x_sol)
    R_sym = sp.simplify(R_poly).subs(t, t - t_offset)

    Vx_sym = sp.diff(R_sym, t)
    Ax_sym = sp.diff(Vx_sym, t)


    # -----------------------------
    # Vertical Trajectory (11th order)
    # -----------------------------
    H_powers = [t**i for i in range(12)]
    y_coeffs = sp.symbols('y0:12')
    H_poly = sum(c*p for c,p in zip(y_coeffs,H_powers))

    Vy_poly = sp.diff(H_poly,t)
    Ay_poly = sp.diff(Vy_poly,t)
    Jy_poly = sp.diff(Ay_poly,t)
    Sy_poly = sp.diff(Jy_poly,t)
    Cy_poly = sp.diff(Sy_poly,t)

    H_eqs = [
        H_poly.subs(t,0) - H_start,
        H_poly.subs(t,t1) - H_end,
        Vy_poly.subs(t,0),
        Vy_poly.subs(t,t1),
        Ay_poly.subs(t,0),
        Ay_poly.subs(t,t1),
        Jy_poly.subs(t,0),
        Jy_poly.subs(t,t1),
        Sy_poly.subs(t,0),
        Sy_poly.subs(t,t1),
        Cy_poly.subs(t,0),
        Cy_poly.subs(t,t1)
    ]

    H_mat = sp.linear_eq_to_matrix(H_eqs, y_coeffs)
    y_sol = sp.solve_linear_system(H_mat[0].row_join(sp.Matrix(H_mat[1])), *y_coeffs)

    H_poly = H_poly.subs(y_sol)
    H_sym = sp.simplify(H_poly).subs(t, t - t_offset)

    Vy_sym = sp.diff(H_sym, t)
    Ay_sym = sp.diff(Vy_sym, t)

    return H_sym, R_sym, Vx_sym, Vy_sym, Ax_sym, Ay_sym, t

def get_thrust(H_sym, Vx_sym, Vy_sym, Ax_sym, Ay_sym, rho_0, M, g, Cd, Cl, Ab_frontal, Aw_planform, Ab_top):
    """
    Generates horizontal and vertical thrust profiles Tx(t) & Ty(t)
    for a given aircraft trajectory using symbolic expressions.

    Inputs:
        H_sym       : altitude function H(t) [m] (SymPy expression)
        Vx_sym      : horizontal velocity Vx(t) [m/s] (SymPy expression)
        Vy_sym      : vertical velocity Vy(t) [m/s] (SymPy expression)
        Ax_sym      : horizontal acceleration Ax(t) [m/s²] (SymPy expression)
        Ay_sym      : vertical acceleration Ay(t) [m/s²] (SymPy expression)
        rho_0       : air density at sea level [kg/m³]
        M           : aircraft mass [kg]
        Cd          : drag coefficient (dimensionless)
        Cl          : lift coefficient (dimensionless)
        Ab_frontal  : aircraft frontal area [m²]
        Aw_planform : wing planform area [m²]
        Ab_top      : top surface area of aircraft [m²]

    Outputs:
        Tx_sym : horizontal thrust [N] (SymPy expression)
        Ty_sym : vertical thrust [N] (SymPy expression)
    """

    rho_sym = rho_0 * sp.exp(-H_sym / 8500)

    Tx_sym = (M * Ax_sym) + (0.5 * Cd * rho_sym * Ab_frontal * (Vx_sym**2))
    Ty_sym = M * (Ay_sym + g) - 0.5* rho_sym * (Cl * Aw_planform * (Vx_sym**2) - Cd * Ab_top * (Vy_sym**2))

    return Tx_sym, Ty_sym

def get_power(Tx_sym, Ty_sym, H_sym, Vx_sym, Vy_sym, A_rotor, rotor_count, rho_0):
    """
    Computes total rotor power required for a given trajectory, 
    using horizontal and vertical thrust and aircraft kinematics.

    Inputs:
        Tx_sym      : horizontal thrust [N] (SymPy expression)
        Ty_sym      : vertical thrust [N] (SymPy expression)
        H_sym       : altitude H(t) [m] (SymPy expression)
        Vx_sym      : horizontal velocity Vx(t) [m/s] (SymPy expression)
        Vy_sym      : vertical velocity Vy(t) [m/s] (SymPy expression)
        A_rotor     : single rotor disk area [m²]
        rotor_count : number of rotors

    Outputs:
        T_sym       : total rotor thrust [N] (SymPy expression)
        P_sym       : total power [W] (SymPy expression)
    """

    T_sym = sp.sqrt(Tx_sym**2 + Ty_sym**2)
    T_rotor = T_sym/rotor_count
    
    rho_sym = rho_0*sp.exp(-H_sym/8500)
    vh = sp.sqrt(T_rotor / (2 * rho_sym * (A_rotor))) # Hover induced velocity

    # vi = T_rotor / (2 * rho * (A_rotor) * sqrt(V**2 + vh**2))
    V = sp.sqrt(Vx_sym**2 + Vy_sym**2)
    vi = sp.sqrt((sp.sqrt(V**4 + 4*vh**4) - V**2)/2)

    p_useful = (Tx_sym * Vx_sym + Ty_sym * Vy_sym)/rotor_count

    kappa = 1.15
    p_induced = (kappa * T_rotor * vi)

    # Total Power
    P_sym = (p_useful + p_induced)*rotor_count

    return T_sym, P_sym

def get_rotor_tilt(Vx_sym, Vy_sym, Tx_sym, Ty_sym):
    t = sp.symbols('t')
    epsilon = 1e-6  # small number to avoid divide-by-zero
    # Flight path angle: 0 to 90°
    flight_angle_sym = sp.Piecewise(
        (sp.pi/2, sp.Abs(Vx_sym) < epsilon),  # Vx ≈ 0 → angle = π/2
        (sp.atan2(sp.Abs(Vy_sym), Vx_sym), True))
    tilt_angle_sym = sp.atan2(Tx_sym, Ty_sym)
    tilt_speed_sym = sp.diff(tilt_angle_sym, t)
    tilt_acc_sym = sp.diff(tilt_speed_sym, t)

    return flight_angle_sym, tilt_angle_sym, tilt_speed_sym, tilt_acc_sym

def eval_sym(exprs, t, time):
    return [np.full_like(time, sp.lambdify(t, e, "numpy")(time), dtype=float)
            if np.isscalar(sp.lambdify(t, e, "numpy")(time))
            else np.array(sp.lambdify(t, e, "numpy")(time), dtype=float)
            for e in exprs]

def phases(cl, cr, ds, t, t_climb, t_cruise):
    return sp.Piecewise(
        (cl, t <= t_climb),
        (cr, t <= t_climb + t_cruise),
        (ds, True)
    )

