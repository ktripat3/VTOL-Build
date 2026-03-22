import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt


def get_flight_params(mission, aircraft):
    # =============================
    # Constants
    # =============================
    rho_0 = 1.225               # Air Density @ Sea Level (kg/m^3)
    g = 9.81                    # Acceleration due to gravity (m/s^2)

    # =============================
    # Flight Parameters
    # =============================
    M = mission[0]
    A = mission[1]

    Aw_planform = aircraft[0]
    Ab_frontal = aircraft[1]
    A_rotor = np.pi * (aircraft[2]/2)**2
    rotor_count = aircraft[3]
    Cd = aircraft[4]
    Cl = aircraft[5]
    rho_A = rho_0 * np.exp(-A/8500)
    Vc = np.sqrt(2*M*g/(Cl*rho_A*Aw_planform))
    print(Vc)
    
    flight_params = [rho_0, g, M, A, Vc, Aw_planform, Ab_frontal, A_rotor, rotor_count, Cd, Cl]

    return flight_params

def get_phase_trajectory(profile_params):
    H_start, H_end, R_start, Vx_start, Vx_end, t_start, t_end = profile_params

    t = np.linspace(t_start, t_end, 1000)
    t1 = t_end - t_start

    # =============================
    # Horizontal (6th order)
    # =============================
    def poly_row(t): return np.array([t**i for i in range(7)])
    def d1_row(t): return np.array([0] + [i * t**(i-1) for i in range(1, 7)])
    def d2_row(t): return np.array([0, 0] + [i*(i-1)*t**(i-2) for i in range(2, 7)])
    def d3_row(t): return np.array([0, 0, 0] + [i*(i-1)*(i-2)*t**(i-3) for i in range(3, 7)])
    

    A = np.zeros((7, 7))
    b = np.zeros(7)

    A[0] = poly_row(0);    b[0] = R_start
    A[1] = d1_row(0);      b[1] = Vx_start
    A[2] = d1_row(t1);     b[2] = Vx_end
    A[3] = d2_row(0);      b[3] = 0
    A[4] = d2_row(t1);     b[4] = 0
    A[5] = d3_row(0);      b[5] = 0
    A[6] = d3_row(t1);     b[6] = 0

    x_coeffs = np.linalg.solve(A, b)

    # =============================
    # Vertical (11th order)
    # =============================
    def d1_row_11(t): return np.array([0] + [i * t**(i-1) for i in range(1, 12)])
    def d2_row_11(t): return np.array([0, 0] + [i*(i-1)*t**(i-2) for i in range(2, 12)])
    def d3_row_11(t): return np.array([0, 0, 0] + [i*(i-1)*(i-2)*t**(i-3) for i in range(3, 12)])
    def d4_row_11(t): return np.array([0, 0, 0, 0] + [i*(i-1)*(i-2)*(i-3)*t**(i-4) for i in range(4, 12)])
    def d5_row_11(t): return np.array([0, 0, 0, 0, 0] + [i*(i-1)*(i-2)*(i-3)*(i-4)*t**(i-5) for i in range(5, 12)])
    def poly_row_11(t): return np.array([t**i for i in range(12)])

    A = np.zeros((12, 12))
    b = np.zeros(12)

    A[0]  = poly_row_11(0);   b[0]  = H_start
    A[1]  = poly_row_11(t1);  b[1]  = H_end
    A[2]  = d1_row_11(0);     b[2]  = 0
    A[3]  = d1_row_11(t1);    b[3]  = 0
    A[4]  = d2_row_11(0);     b[4]  = 0
    A[5]  = d2_row_11(t1);    b[5]  = 0
    A[6]  = d3_row_11(0);     b[6]  = 0
    A[7]  = d3_row_11(t1);    b[7]  = 0
    A[8]  = d4_row_11(0);     b[8]  = 0
    A[9]  = d4_row_11(t1);    b[9]  = 0
    A[10] = d5_row_11(0);     b[10] = 0
    A[11] = d5_row_11(t1);    b[11] = 0

    y_coeffs = np.linalg.solve(A, b)

    # =============================
    # Shift time (t - t_start)
    # =============================
    tau = t - t_start

    # =============================
    # Evaluate polynomials
    # =============================
    R = np.polyval(x_coeffs[::-1], tau)
    H = np.polyval(y_coeffs[::-1], tau)

    # =============================
    # Derivatives via coefficients
    # =============================
    def poly_derivative(coeffs):
        return np.array([i * coeffs[i] for i in range(1, len(coeffs))])

    dx = poly_derivative(x_coeffs)
    ddx = poly_derivative(dx)

    dy = poly_derivative(y_coeffs)
    ddy = poly_derivative(dy)

    Vx = np.polyval(dx[::-1], tau)
    Ax = np.polyval(ddx[::-1], tau)

    Vy = np.polyval(dy[::-1], tau)
    Ay = np.polyval(ddy[::-1], tau)

    # =============================
    # Flight Path Angle (degrees)
    # =============================\
    epsilon = 1e-6
    gamma = np.arctan2(Vy, np.where(np.abs(Vx) < epsilon, epsilon, Vx))
    
    trajectory = [H, R, Vx, Vy, Ax, Ay, gamma, t]
    return trajectory

def get_thrust(flight_params, trajectory):
    rho_0, g, M, A, Vc, Aw_planform, Ab_frontal, A_rotor, rotor_count, Cd, Cl = flight_params
    H, R, Vx, Vy, Ax, Ay, gamma, t = trajectory

    V = np.sqrt(Vx**2 + Vy**2)
    rho = rho_0 * np.exp(-H / 8500)
    
    temp = (0.5 * rho * (V**2))
    Tx = (M * Ax) + temp*((Cd * Ab_frontal * np.cos(gamma)) + (Cl * Aw_planform * np.sin(gamma)))
    Ty = M * (Ay + g) + temp*((Cd * Ab_frontal * np.sin(gamma)) - (Cl * Aw_planform * np.cos(gamma)))
    # Tx_down = (M * Ax) + temp*((Cd * Ab_frontal * np.cos(gamma)) - (Cl * Aw_planform * np.sin(gamma)))
    # Ty_down = M * (Ay + g) - temp*((Cd * Ab_frontal * np.sin(gamma)) + (Cl * Aw_planform * np.cos(gamma)))

    # Tx = np.where(Vy >= 0, Tx_up, Tx_down)
    # Ty = np.where(Vy >= 0, Ty_up, Ty_down)
    T = np.sqrt(Tx**2 + Ty**2)
    T_rotor = T/rotor_count
    thrust = [Tx, Ty, T, T_rotor]

    return thrust

def get_power(flight_params, trajectory, thrust):
    rho_0, g, M, A, Vc, Aw_planform, Ab_frontal, A_rotor, rotor_count, Cd, Cl = flight_params
    H, R, Vx, Vy, Ax, Ay, gamma, t = trajectory
    Tx, Ty, T, T_rotor = thrust

    V = np.sqrt(Vx**2 + Vy**2)
    rho = rho_0 * np.exp(-H / 8500)

    vh = np.sqrt(T_rotor / (2 * rho * A_rotor)) # Hover induced velocity
    vi = np.sqrt((np.sqrt(V**4 + 4*vh**4) - V**2)/2)
    P_useful = T_rotor * V
    kappa = 1.15
    P_induced = (kappa * T_rotor * vi)
    
    P_rotor = P_useful + P_induced
    P = P_rotor*rotor_count
    E = cumulative_trapezoid(P, t, initial=0)
    power = [E, P, P_rotor]
    
    return power

def get_rotor_tilt(trajectory, thrust):
    H, R, Vx, Vy, Ax, Ay, gamma, t = trajectory
    Tx, Ty, T, T_rotor = thrust

    dt = t[1] - t[0]
    rotor_tilt_angle = np.arctan2(Tx, Ty)
    rotor_tilt_speed = np.gradient(rotor_tilt_angle, t)
    rotor_tilt_acc = np.gradient(rotor_tilt_speed, t)

    rotor_tilt = [rotor_tilt_angle, rotor_tilt_speed, rotor_tilt_acc]
    return rotor_tilt

def get_phases(flight_params, phase_times):
    rho_0, g, M, A, Vc, Aw_planform, Ab_frontal, A_rotor, rotor_count, Cd, Cl = flight_params
    t_climb, t_cruise, t_descent = phase_times

    phases = ['climb', 'cruise', 'descent']
    phase_params_dict = {
        'climb':   [0, A, 0, 0, Vc, 0, t_climb],
        'cruise':  [A, A, 0, Vc, Vc, 0, t_cruise],
        'descent': [A, 0, 0, Vc, 0, 0, t_descent]
    }
    return phases, phase_params_dict

# =============================
# Directly Executable functions
# =============================

def solve_phase(mission, aircraft, profile_params):
    flight_params = get_flight_params(mission, aircraft)
    trajectory = get_phase_trajectory(profile_params)
    thrust = get_thrust(flight_params, trajectory)
    power = get_power(flight_params, trajectory, thrust)
    rotor_tilt = get_rotor_tilt(trajectory, thrust)
    phase_results = [trajectory, thrust, power, rotor_tilt]
    
    H, R, Vx, Vy, Ax, Ay, gamma, t = trajectory
    Tx, Ty, T, T_rotor = thrust
    E, P, P_rotor = power

    T_rotor_max = np.max(T_rotor)/1000
    P_rotor_max = np.max(P_rotor)/1000
    Range = R[-1]/1000
    Energy = E[-1]/3.6E6
    unit_range = Range / Energy

    performance = [T_rotor_max, P_rotor_max, Range, Energy, unit_range]

    return phase_results, performance

def print_performance_metrics(performance):
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

    print(bcolors.BOLD + bcolors.HEADER + "\n------------------------ Flight Metrics ------------------------" + bcolors.END)
    print(f"{bcolors.OKGREEN}Total Range: {bcolors.END}{round(performance[2]):,} km")
    print(f"{bcolors.OKBLUE}Peak Thrust per Rotor: {bcolors.END}{round(performance[0]):,} kN")
    print(f"{bcolors.OKCYAN}Peak Power per Rotor: {bcolors.END}{round(performance[1]):,} kW")
    print(f"{bcolors.WARNING}Total Energy Consumption: {bcolors.END}{round(performance[3]):,} kWh")
    print(f"{bcolors.BOLD}Range per Unit Energy Consumption: {bcolors.END}{round(performance[4],3):,} km/kWh")
    print(bcolors.BOLD + bcolors.HEADER + "---------------------------------------------------------------\n" + bcolors.END)
    
    return

def plot_time_trajectory(phase_results, phase_name):
    trajectory, *_, rotor_tilt = phase_results
    H, R, Vx, Vy, Ax, Ay, gamma, t = trajectory
    tilt_angle, tilt_speed, tilt_acc = rotor_tilt

    fig, axes = plt.subplots(6, 3, figsize=(15,12))

    # Grayscale palette
    primary_color   = '#d9885f'  # dull/muted orange
    secondary_color = '#7fa97f'  # dull/muted green
    tertiary_color  = '#dcdcdc'  # light gray for grids
    
    # Create 6-row figure
    fig, axes = plt.subplots(6, 1, figsize=(10, 12))

    # --- 1. Ground Range & Altitude ---
    ax1 = axes[0]
    ax1.plot(t, R/1000, color=primary_color, label="Ground Range")
    ax1.set_ylabel("Range [km]", color=primary_color)
    ax1.tick_params(axis='y', labelcolor=primary_color)
    ax1.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    ax1b = ax1.twinx()
    ax1b.plot(t, H, color=secondary_color, label="Altitude")
    ax1b.set_ylabel("Altitude [m]", color=secondary_color)
    ax1b.tick_params(axis='y', labelcolor=secondary_color)
    ax1.set_title(phase_name, fontsize=12, fontweight='bold')

    # --- 2. Flight Path Angle & Tilt Angle ---
    ax2 = axes[1]
    ax2.plot(t, np.degrees(gamma), color=primary_color, label="Flight Path Angle")
    ax2.set_ylabel("Flight Path [deg]", color=primary_color)
    ax2.tick_params(axis='y', labelcolor=primary_color)
    ax2.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    ax2b = ax2.twinx()
    ax2b.plot(t, np.degrees(tilt_angle), color=secondary_color, label="Tilt Angle")
    ax2b.set_ylabel("Tilt [deg]", color=secondary_color)
    ax2b.tick_params(axis='y', labelcolor=secondary_color)

    # --- 3. Velocities ---
    ax3 = axes[2]
    ax3.plot(t, Vx, color=primary_color, label="Vx")
    ax3.set_ylabel("Vx [m/s]", color=primary_color)
    ax3.tick_params(axis='y', labelcolor=primary_color)
    ax3.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    ax3b = ax3.twinx()
    ax3b.plot(t, Vy, color=secondary_color, label="Vy")
    ax3b.set_ylabel("Vy [m/s]", color=secondary_color)
    ax3b.tick_params(axis='y', labelcolor=secondary_color)

    # --- 4. Tilt Speed ---
    axes[3].plot(t, np.degrees(tilt_speed), color='black', label="Tilt Speed")
    axes[3].set_ylabel("Tilt Speed [deg/s]")
    axes[3].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    # --- 5. Accelerations ---
    ax5 = axes[4]
    ax5.plot(t, Ax, color=primary_color, label="Ax")
    ax5.set_ylabel("Ax [m/s²]", color=primary_color)
    ax5.tick_params(axis='y', labelcolor=primary_color)
    ax5.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    ax5b = ax5.twinx()
    ax5b.plot(t, Ay, color=secondary_color, label="Ay")
    ax5b.set_ylabel("Ay [m/s²]", color=secondary_color)
    ax5b.tick_params(axis='y', labelcolor=secondary_color)

    # --- 6. Tilt Acceleration ---
    axes[5].plot(t, np.degrees(tilt_acc), color='black', label="Tilt Acceleration")
    axes[5].set_ylabel("Tilt Acc [deg/s²]")
    axes[5].set_xlabel("Time [s]")
    axes[5].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_range_trajectory(phase_results, phase_name):
    trajectory, _, _, rotor_tilt = phase_results
    H, R, Vx, Vy, Ax, Ay, gamma, t = trajectory
    tilt_angle, tilt_speed, tilt_acc = rotor_tilt

    # Grayscale palette
    primary_color   = '#d9885f'  # dull/muted orange
    secondary_color = '#7fa97f'  # dull/muted green
    tertiary_color  = '#dcdcdc'  # light gray for grids
    fig, axes = plt.subplots(6, 1, figsize=(10, 12))

    # --- 1. Altitude vs Range ---
    axes[0].plot(R/1000, H, color='black', linestyle='-')
    axes[0].set_ylabel("Altitude [m]", color=primary_color)
    axes[0].tick_params(axis='y', labelcolor=primary_color)
    axes[0].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)
    axes[0].set_title(phase_name, fontsize=12, fontweight='bold')

    # --- 2. Flight Path & Tilt Angle ---
    axes[1].plot(R/1000, np.degrees(gamma), color=primary_color, label="Flight Path Angle")
    axes[1].set_ylabel("Flight Path [deg]", color=primary_color)
    axes[1].tick_params(axis='y', labelcolor=primary_color)
    axes[1].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    ax2b = axes[1].twinx()
    ax2b.plot(R/1000, np.degrees(tilt_angle), color=secondary_color, label="Tilt Angle")
    ax2b.set_ylabel("Tilt [deg]", color=secondary_color)
    ax2b.tick_params(axis='y', labelcolor=secondary_color)

    # --- 3. Vx & Vy ---
    axes[2].plot(R/1000, Vx, color=primary_color, label="Vx")
    axes[2].set_ylabel("Vx [m/s]", color=primary_color)
    axes[2].tick_params(axis='y', labelcolor=primary_color)
    axes[2].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    ax3b = axes[2].twinx()
    ax3b.plot(R/1000, Vy, color=secondary_color, label="Vy")
    ax3b.set_ylabel("Vy [m/s]", color=secondary_color)
    ax3b.tick_params(axis='y', labelcolor=secondary_color)

    # --- 4. Tilt Speed ---
    axes[3].plot(R/1000, np.degrees(tilt_speed), color='black', label="Tilt Speed")
    axes[3].set_ylabel("Tilt Speed [deg/s]")
    axes[3].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    # --- 5. Ax & Ay ---
    axes[4].plot(R/1000, Ax, color=primary_color, label="Ax")
    axes[4].set_ylabel("Ax [m/s²]", color=primary_color)
    axes[4].tick_params(axis='y', labelcolor=primary_color)
    axes[4].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    ax5b = axes[4].twinx()
    ax5b.plot(R/1000, Ay, color=secondary_color, label="Ay")
    ax5b.set_ylabel("Ay [m/s²]", color=secondary_color)
    ax5b.tick_params(axis='y', labelcolor=secondary_color)

    # --- 6. Tilt Acceleration ---
    axes[5].plot(R/1000, np.degrees(tilt_acc), color='black', label="Tilt Acceleration")
    axes[5].set_ylabel("Tilt Acc [deg/s²]")
    axes[5].set_xlabel("Range [km]")
    axes[5].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_energy_req(mission, phase_results):
    trajectory, thrust, power, rotor_tilt = phase_results
    H, R, Vx, Vy, Ax, Ay, gamma, t = trajectory
    Tx, Ty, T, T_rotor = thrust
    E, P, P_rotor = power

    # Grayscale palette
    primary_color   = '#d9885f'  # dull/muted orange
    secondary_color = '#7fa97f'  # dull/muted green
    tertiary_color  = '#dcdcdc'  # light gray for grids

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
    ln3 = ax2.plot(R/1000, (mission[2]/3.6e6)*np.ones_like(R), color='red', linewidth=1.5, linestyle=':', label="70 % Capacity")
    ln4 = ax2.plot(R/1000, ((mission[2]/(1-mission[3]))/3.6e6)*np.ones_like(R), color='red', linewidth=1.5, linestyle='-.', label="Total Capacity")

    ax2.set_ylabel("Energy [kWh]", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combined legend
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')

    plt.tight_layout()
    plt.show()

