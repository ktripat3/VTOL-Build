import sympy as sp
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

def get_params(mission, aircraft):
    # Constants
    rho_0 = 1.225               # Air Density @ Sea Level (kg/m^3)
    g = 9.81                    # Acceleration due to gravity (m/s^2)

    # Flight Parameters
    M = mission[0]
    A = mission[1]

    Aw_planform = aircraft[0]
    Ab_frontal = aircraft[1]
    A_rotor = sp.pi * (aircraft[2]/2)**2
    Cd = aircraft[3]
    Cl = aircraft[4]
    rho_A = rho_0 * sp.exp(-A/8500)
    Vc = sp.sqrt(2*M*g/(Cl*rho_A*Aw_planform))
    
    return rho_0, g, M, A, Vc, Aw_planform, Ab_frontal, A_rotor, Cd, Cl

def get_phase_trajectory(H_start, H_end, R_start, Vx_start, Vx_end, t_phase, t_offset):
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

def get_thrust(H_sym, Vx_sym, Vy_sym, Ax_sym, Ay_sym, rho_0, M, g, Cd, Cl, Ab_frontal, Aw_planform):

    rho_sym = rho_0 * sp.exp(-H_sym / 8500)

    V = sp.sqrt(Vx_sym**2 + Vy_sym**2)
    gamma = sp.atan2(Vy_sym,Vx_sym)

    Tx_up = (M * Ax_sym) + (0.5 * rho_sym * (V**2)) * ((Cd * Ab_frontal * sp.cos(gamma)) + (Cl * Aw_planform * sp.sin(gamma)))
    Ty_up = M * (Ay_sym + g) + (0.5 * rho_sym * (V**2)) * ((Cd * Ab_frontal * sp.sin(gamma)) - (Cl * Aw_planform * sp.cos(gamma)))
    Tx_down = (M * Ax_sym) + (0.5 * rho_sym * (V**2)) * ((Cd * Ab_frontal * sp.cos(gamma)) - (Cl * Aw_planform * sp.sin(gamma)))
    Ty_down = M * (Ay_sym + g) - (0.5 * rho_sym * (V**2)) * ((Cd * Ab_frontal * sp.sin(gamma)) + (Cl * Aw_planform * sp.cos(gamma)))

    Tx_sym = sp.Piecewise((Tx_up, Vy_sym >= 0),(Tx_down, True))
    Ty_sym = sp.Piecewise((Ty_up, Vy_sym >= 0),(Ty_down, True))

    return Tx_sym, Ty_sym

def get_power(Tx_sym, Ty_sym, H_sym, Vx_sym, Vy_sym, A_rotor, rotor_count, rho_0):

    T_sym = sp.sqrt(Tx_sym**2 + Ty_sym**2)
    T_rotor = T_sym/rotor_count
    
    rho_sym = rho_0*sp.exp(-H_sym/8500)
    vh = sp.sqrt(T_rotor / (2 * rho_sym * (A_rotor))) # Hover induced velocity

    # vi = T_rotor / (2 * rho * (A_rotor) * sqrt(V**2 + vh**2))
    V = sp.sqrt(Vx_sym**2 + Vy_sym**2)
    vi = sp.sqrt((sp.sqrt(V**4 + 4*vh**4) - V**2)/2)

    # p_useful = (Tx_sym * Vx_sym + sp.Max(Ty_sym * Vy_sym, 0)) / rotor_count
    p_useful = T_rotor * V

    kappa = 1.15
    p_induced = (kappa * T_rotor * vi)

    # Total Power
    P_sym = (p_useful + p_induced)*rotor_count

    return T_sym, P_sym

def get_tilt(Vx_sym, Vy_sym, Tx_sym, Ty_sym):
    t = sp.symbols('t')
    flight_path_sym = sp.atan2(Vy_sym, Vx_sym)
    tilt_angle_sym = sp.atan2(Tx_sym, Ty_sym)
    tilt_speed_sym = sp.diff(tilt_angle_sym, t)
    tilt_acc_sym = sp.diff(tilt_speed_sym, t)

    return flight_path_sym, tilt_angle_sym, tilt_speed_sym, tilt_acc_sym

def eval_sym(exprs, t, time):
    result = []
    for e in exprs:
        f = sp.lambdify(t, e, "numpy")
        val = f(time)   # return vector, not scalar
        val = np.array(val, dtype=float)
        if val.shape == ():  # if scalar returned
            val = np.full_like(time, val, dtype=float)
        result.append(val)
    return result

def solve_phase(H_start, H_end, R_start, Vx_start, Vx_end,
                t_phase, t_offset, time_vec,
                rho_0, M, g, Cd, Cl, Ab_frontal, Aw_planform,
                A_rotor, rotor_count):

    # -----------------------------
    # Trajectory
    # -----------------------------
    H_sym, R_sym, Vx_sym, Vy_sym, Ax_sym, Ay_sym, t = get_phase_trajectory(
        H_start, H_end, R_start, Vx_start, Vx_end, t_phase, t_offset
    )

    H, R, Vx, Vy, Ax, Ay = eval_sym(
        [H_sym, R_sym, Vx_sym, Vy_sym, Ax_sym, Ay_sym],
        t, time_vec
    )

    # -----------------------------
    # Thrust
    # -----------------------------
    Tx_sym, Ty_sym = get_thrust(
        H_sym, Vx_sym, Vy_sym, Ax_sym, Ay_sym,
        rho_0, M, g, Cd, Cl, Ab_frontal, Aw_planform
    )

    Tx, Ty = eval_sym([Tx_sym, Ty_sym], t, time_vec)

    # -----------------------------
    # Power
    # -----------------------------
    T_sym, P_sym = get_power(Tx_sym, Ty_sym, H_sym, Vx_sym, Vy_sym, A_rotor, rotor_count, rho_0)

    T, P = eval_sym([T_sym, P_sym], t, time_vec)

    # -----------------------------
    # Tilt / Flight Path
    # -----------------------------
    flight_sym, tilt_sym, tilt_speed_sym, tilt_acc_sym = get_tilt(
        Vx_sym, Vy_sym, Tx_sym, Ty_sym
    )

    flight, tilt, tilt_speed, tilt_acc = eval_sym(
        [flight_sym, tilt_sym, tilt_speed_sym, tilt_acc_sym],
        t, time_vec
    )

    return {
        "H": H,
        "R": R,
        "Vx": Vx,
        "Vy": Vy,
        "Ax": Ax,
        "Ay": Ay,
        "Tx": Tx,
        "Ty": Ty,
        "T": T,
        "P": P,
        "flight_angle": flight,
        "tilt_angle": tilt,
        "tilt_speed": tilt_speed,
        "tilt_acc": tilt_acc
    }

def get_controls(mission, aircraft):

    t_cl = mission[3]
    t_cr = mission[4]
    t_ds = mission[5]
    rotor_count = aircraft[4]

    rho_0, g, M, A, Vc, Aw_planform, Ab_frontal, A_rotor, Cd, Cl = get_params(mission, aircraft)

    # Time Domains for Climb, Cruise, Descent
    t1 = np.linspace(0, t_cl, 2*t_cl)
    t2 = np.linspace(t_cl, t_cl+t_cr, 2*t_cr)
    t3 = np.linspace(t_cl+t_cr, t_cl+t_cr+t_ds, 2*t_ds)
    t_phases = [t1, t2, t3]
    # -----------------------------
    # Compute Flight Controls
    # -----------------------------
    # Climb Profiles
    cl = solve_phase(0, A, 0, 0, Vc, t_cl, 0, t1, rho_0, M, g, Cd, Cl, Ab_frontal, Aw_planform, A_rotor, rotor_count)

    # Cruise Profiles
    cr = solve_phase(A, A, cl["R"][-1], Vc, Vc, t_cr, t_cl, t2, rho_0, M, g, Cd, Cl, Ab_frontal, Aw_planform, A_rotor, rotor_count)

    # Descent Profiles
    ds = solve_phase(A, 0, cr["R"][-1], Vc, 0, t_ds, t_cl + t_cr, t3, rho_0, M, g, Cd, Cl, Ab_frontal, Aw_planform, A_rotor, rotor_count)

    # -----------------------------
    # Compile Profiles
    # -----------------------------
    flight = {"Climb": cl, "Cruise": cr, "Descent": ds}
    flight_phases = list(flight.keys())
    phases = list(flight.values())

    H_phases = [p["H"] for p in phases]
    R_phases = [p["R"] for p in phases]

    Vx_phases = [p["Vx"] for p in phases]
    Vy_phases = [p["Vy"] for p in phases]

    Ax_phases = [p["Ax"] for p in phases]
    Ay_phases = [p["Ay"] for p in phases]

    Tx_phases = [p["Tx"] for p in phases]
    Ty_phases = [p["Ty"] for p in phases]

    T_phases = [p["T"] for p in phases]
    P_phases = [p["P"] for p in phases]
    P = np.concatenate(P_phases)
    time = np.concatenate(t_phases)
    if np.isnan(P[0]):
        P[0] = P[1]
    E = cumulative_trapezoid(P, time, initial=0)

    flight_angle_phases = [p["flight_angle"] for p in phases]
    flight_angle_phases[-1][-1] = 0

    tilt_angle_phases = [p["tilt_angle"] for p in phases]
    tilt_speed_phases = [p["tilt_speed"] for p in phases]
    tilt_acc_phases = [p["tilt_acc"] for p in phases]

    return flight_phases, t_phases, H_phases, R_phases, Vx_phases, Vy_phases, Ax_phases, Ay_phases, Tx_phases, Ty_phases, T_phases, P_phases, E, flight_angle_phases, tilt_angle_phases, tilt_speed_phases, tilt_acc_phases

def get_performance(controls):
    _, _, _, R_phases, *_, E, _, _, _, _ = controls
    # Performance
    Range = R_phases[-1][-1]
    Energy = E[-1]
    unit_range = 3600 * Range / Energy
    print(f"Range per unit Energy Consumption = {unit_range} km/kWh")
    return unit_range

def plot_time_trajectory(controls):

    flight_phases, t_phases, H_phases, R_phases, Vx_phases, Vy_phases, Ax_phases, Ay_phases, *_, flight_angle_phases, tilt_angle_phases, tilt_speed_phases, tilt_acc_phases = controls
    fig, axes = plt.subplots(6, 3, figsize=(15,12))

    # Grayscale palette
    primary_color   = '#d9885f'  # dull/muted orange
    secondary_color = '#7fa97f'  # dull/muted green
    tertiary_color  = '#dcdcdc'  # light gray for grids

    for i in range(3):

        # 1. Ground Range & Altitude
        ax1 = axes[0,i]
        ax1.plot(t_phases[i], R_phases[i]/1000, color=primary_color, linestyle='-', label='Ground Range')
        ax1.set_title(flight_phases[i], fontsize=12, fontweight='bold')
        ax1.set_ylabel("Range [km]", color=primary_color)
        ax1.tick_params(axis='y', labelcolor=primary_color)
        ax1.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

        ax1b = ax1.twinx()
        ax1b.plot(t_phases[i], H_phases[i], color=secondary_color, linestyle='-', label='Altitude')
        ax1b.set_ylabel("Altitude [m]", color=secondary_color)
        ax1b.tick_params(axis='y', labelcolor=secondary_color)

        # 2. Flight Path Angle & Tilt Angle
        ax2 = axes[1,i]
        ax2.plot(t_phases[i], (180/np.pi)*flight_angle_phases[i], color=primary_color, linestyle='-', label='Flight Path')
        ax2.set_ylabel("Flight Path [deg]", color=primary_color)
        ax2.tick_params(axis='y', labelcolor=primary_color)
        ax2.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

        ax2b = ax2.twinx()
        ax2b.plot(t_phases[i], (180/np.pi)*tilt_angle_phases[i], color=secondary_color, linestyle='-', label='Tilt Angle')
        ax2b.set_ylabel("Tilt [deg]", color=secondary_color)
        ax2b.tick_params(axis='y', labelcolor=secondary_color)

        # 3. Vx & Vy
        ax3 = axes[2,i]
        ax3.plot(t_phases[i], Vx_phases[i], color=primary_color, linestyle='-', label='Vx')
        ax3.set_ylabel("Vx [m/s]", color=primary_color)
        ax3.tick_params(axis='y', labelcolor=primary_color)
        ax3.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

        ax3b = ax3.twinx()
        ax3b.plot(t_phases[i], Vy_phases[i], color=secondary_color, linestyle='-', label='Vy')
        ax3b.set_ylabel("Vy [m/s]", color=secondary_color)
        ax3b.tick_params(axis='y', labelcolor=secondary_color)

        # 4. Tilt Speed
        axes[3,i].plot(t_phases[i], (180/np.pi)*tilt_speed_phases[i], color='black', linestyle='-')
        axes[3,i].set_ylabel("Tilt Speed [deg/s]")
        axes[3,i].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

        # 5. Ax & Ay
        ax5 = axes[4,i]
        ax5.plot(t_phases[i], Ax_phases[i], color=primary_color, linestyle='-', label='Ax')
        ax5.set_ylabel("Ax [m/s²]", color=primary_color)
        ax5.tick_params(axis='y', labelcolor=primary_color)
        ax5.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

        ax5b = ax5.twinx()
        ax5b.plot(t_phases[i], Ay_phases[i], color=secondary_color, linestyle='-', label='Ay')
        ax5b.set_ylabel("Ay [m/s²]", color=secondary_color)
        ax5b.tick_params(axis='y', labelcolor=secondary_color)

        # 6. Tilt Acceleration
        axes[5,i].plot(t_phases[i], (180/np.pi)*tilt_acc_phases[i], color='black', linestyle='-')
        axes[5,i].set_ylabel("Tilt Acc [deg/s²]")
        axes[5,i].set_xlabel("Time [s]")
        axes[5,i].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_range_trajectory(controls):

    flight_phases, t_phases, H_phases, R_phases, Vx_phases, Vy_phases, Ax_phases, Ay_phases, *_, flight_angle_phases, tilt_angle_phases, tilt_speed_phases, tilt_acc_phases = controls
    fig, axes = plt.subplots(6, 3, figsize=(15,12))

    # Grayscale palette
    primary_color   = '#d9885f'  # dull/muted orange
    secondary_color = '#7fa97f'  # dull/muted green
    tertiary_color  = '#dcdcdc'  # light gray for grids

    for i in range(3):

        # 1. Ground Range & Altitude
        ax1 = axes[0,i]
        ax1.plot(R_phases[i]/1000, H_phases[i], color='black', linestyle='-')
        ax1.set_title(flight_phases[i], fontsize=12, fontweight='bold')
        ax1.set_ylabel("Altitude [m]", color=primary_color)
        ax1.tick_params(axis='y', labelcolor=primary_color)
        ax1.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

        # 2. Flight Path Angle & Tilt Angle
        ax2 = axes[1,i]
        ax2.plot(R_phases[i]/1000, (180/np.pi)*flight_angle_phases[i], color=primary_color, linestyle='-', label='Flight Path')
        ax2.set_ylabel("Flight Path [deg]", color=primary_color)
        ax2.tick_params(axis='y', labelcolor=primary_color)
        ax2.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

        ax2b = ax2.twinx()
        ax2b.plot(R_phases[i]/1000, (180/np.pi)*tilt_angle_phases[i], color=secondary_color, linestyle='-', label='Tilt Angle')
        ax2b.set_ylabel("Tilt [deg]", color=secondary_color)
        ax2b.tick_params(axis='y', labelcolor=secondary_color)

        # 3. Vx & Vy
        ax3 = axes[2,i]
        ax3.plot(R_phases[i]/1000, Vx_phases[i], color=primary_color, linestyle='-', label='Vx')
        ax3.set_ylabel("Vx [m/s]", color=primary_color)
        ax3.tick_params(axis='y', labelcolor=primary_color)
        ax3.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

        ax3b = ax3.twinx()
        ax3b.plot(R_phases[i]/1000, Vy_phases[i], color=secondary_color, linestyle='-', label='Vy')
        ax3b.set_ylabel("Vy [m/s]", color=secondary_color)
        ax3b.tick_params(axis='y', labelcolor=secondary_color)

        # 4. Tilt Speed
        axes[3,i].plot(R_phases[i]/1000, (180/np.pi)*tilt_speed_phases[i], color='black', linestyle='-')
        axes[3,i].set_ylabel("Tilt Speed [deg/s]")
        axes[3,i].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

        # 5. Ax & Ay
        ax5 = axes[4,i]
        ax5.plot(R_phases[i]/1000, Ax_phases[i], color=primary_color, linestyle='-', label='Ax')
        ax5.set_ylabel("Ax [m/s²]", color=primary_color)
        ax5.tick_params(axis='y', labelcolor=primary_color)
        ax5.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

        ax5b = ax5.twinx()
        ax5b.plot(R_phases[i]/1000, Ay_phases[i], color=secondary_color, linestyle='-', label='Ay')
        ax5b.set_ylabel("Ay [m/s²]", color=secondary_color)
        ax5b.tick_params(axis='y', labelcolor=secondary_color)

        # 6. Tilt Acceleration
        axes[5,i].plot(R_phases[i]/1000, (180/np.pi)*tilt_acc_phases[i], color='black', linestyle='-')
        axes[5,i].set_ylabel("Tilt Acc [deg/s²]")
        axes[5,i].set_xlabel("Range [km]")
        axes[5,i].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_energy_req(controls, mission):

    # Grayscale palette
    primary_color   = '#d9885f'  # dull/muted orange
    secondary_color = '#7fa97f'  # dull/muted green
    tertiary_color  = '#dcdcdc'  # light gray for grids

    _, _, _, R_phases, *_, Tx_phases, Ty_phases, T_phases, P_phases, E, _, _, _, _ = controls
    E_max = (1 - mission[6]) * mission[2] * 3600000
    R = np.concatenate(R_phases)
    Tx = np.concatenate(Tx_phases)
    Ty = np.concatenate(Ty_phases)
    T = np.concatenate(T_phases)
    P = np.concatenate(P_phases)

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    fig.suptitle("Thrust, Power & Energy", fontsize=16, fontweight='bold')

    # Total Thrust
    axes[0].plot(R/1000, T/1000, color='black', linewidth=3, label='Total Thrust')
    axes[0].plot(R/1000, Tx/1000, color=primary_color, linewidth=1, label='Horizonmtal Thrust')
    axes[0].plot(R/1000, Ty/1000, color=secondary_color, linewidth=1, label='Vertical Thrust')

    axes[0].set_xlabel("Range [km]")
    axes[0].set_ylabel("Thrust [kN]")
    axes[0].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)
    axes[0].legend(loc='best')

    # Power (Left) and Energy (Right)
    ax1 = axes[1]
    ax2 = ax1.twinx()

    # Power [kW]
    ln1 = ax1.plot(R/1000, P/1000, color='tab:blue', linewidth=2, label="Power [kW]")
    ax1.set_xlabel("Range [km]")
    ax1.set_ylabel("Power [kW]", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    # Energy [kWh]
    ln2 = ax2.plot(R/1000, E/3.6e6, color='tab:red', linewidth=2, linestyle='-', label="Energy [kWh]")
    ln3 = ax2.plot(R/1000, (E_max/3.6e6)*np.ones_like(R), color='red', linewidth=1.5, linestyle=':', label="70 % Capacity")
    ln4 = ax2.plot(R/1000, ((E_max/(1-mission[6]))/3.6e6)*np.ones_like(R), color='red', linewidth=1.5, linestyle='-.', label="Total Capacity")

    ax2.set_ylabel("Energy [kWh]", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combined legend
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')

    plt.tight_layout()
    plt.show()


