import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from tool_directory import get_phase_trajectory_11, get_thrust, get_power, get_rotor_tilt, eval_sym, phases

# -----------------------------
# Constants
# -----------------------------
rho_0 = 1.225               # Air Density @ Sea Level (kg/m^3)
g = 9.81                    # Acceleration due to gravity (m/s^2)
reserve_energy_ratio = 0.3  # Ratio of reserve energy to total energy


# -----------------------------
# Mission Parameters
# -----------------------------
gross_weight = 2404     # Gross Aircraft Weight (kg)
cruise_speed = 300      # Cruise speed (km/h)
cruise_altitude = 4000   # Cruise altitude (m)
energy_capacity = 350    # Battery energy (kWh)
t_climb = 600
t_cruise = 3600


# -----------------------------
# Aircraft Parameters
# -----------------------------
wing_drag_coefficient = 0.018
aircraft_frontal_area = 3
aircraft_planform_area = 30
rotor_diameter = 1.4
rotor_count = 6
wing_planform_area = 12


# -----------------------------
M = gross_weight
A = cruise_altitude
Vc = (5/18) * cruise_speed
Aw_planform = wing_planform_area
Ab_frontal = aircraft_frontal_area
Ab_top = aircraft_planform_area
A_rotor = sp.pi * (rotor_diameter/2)**2
Cd = wing_drag_coefficient
rho_A = rho_0 * sp.exp(-A/8500)
Cl = 2*M*g/(rho_A*Aw_planform*Vc**2)
E_max = (1 - reserve_energy_ratio) * energy_capacity * 3600000


# -----------------------------
# Time Domains for Climb, Cruise
# -----------------------------
t1 = np.linspace(0, t_climb, 2*t_climb)
t2 = np.linspace(t_climb, t_climb+t_cruise, 2*t_climb)
t3 = np.linspace(t_cruise+t_climb, t_cruise+t_climb*2, 2*t_climb)
t_total = t_cruise + 2*t_climb
time = np.linspace(0, t_total, 2*t_total)

# -----------------------------
# Compute Flight Controls
# -----------------------------
# Climb Profiles
H_cl_sym, R_cl_sym, Vx_cl_sym, Vy_cl_sym, Ax_cl_sym, Ay_cl_sym, t = get_phase_trajectory_11(0, A, 0, 0, Vc, t_climb, 0)
H_cl, R_cl, Vx_cl, Vy_cl, Ax_cl, Ay_cl = eval_sym([H_cl_sym,R_cl_sym,Vx_cl_sym,Vy_cl_sym,Ax_cl_sym,Ay_cl_sym], t, t1)

# Cruise Profiles
t = sp.symbols('t')
H_cr_sym  = A + 0*t
R_cr_sym  = R_cl_sym.subs(t, t_climb) + Vc*(t - t_climb)
Vx_cr_sym = Vc
Vy_cr_sym = 0
Ax_cr_sym = 0
Ay_cr_sym = 0
H_cr, R_cr, Vx_cr, Vy_cr, Ax_cr, Ay_cr = eval_sym([H_cr_sym,R_cr_sym,Vx_cr_sym,Vy_cr_sym,Ax_cr_sym,Ay_cr_sym], t, t2)

# Descent Profiles
H_ds_sym, R_ds_sym, Vx_ds_sym, Vy_ds_sym, Ax_ds_sym, Ay_ds_sym, t = get_phase_trajectory_11(A, 0, R_cr[-1], Vc, 0, t_climb, t2[-1])
H_ds, R_ds, Vx_ds, Vy_ds, Ax_ds, Ay_ds = eval_sym([H_ds_sym,R_ds_sym,Vx_ds_sym,Vy_ds_sym,Ax_ds_sym,Ay_ds_sym], t, t3)

# Mission Trajectory Profiles
H_sym, R_sym, Vx_sym, Vy_sym, Ax_sym, Ay_sym = [
    phases(cl, cr, ds, t, t_climb, t_cruise)
    for cl, cr, ds in zip(
        [H_cl_sym, R_cl_sym, Vx_cl_sym, Vy_cl_sym, Ax_cl_sym, Ay_cl_sym],
        [H_cr_sym, R_cr_sym, Vx_cr_sym, Vy_cr_sym, Ax_cr_sym, Ay_cr_sym],
        [H_ds_sym, R_ds_sym, Vx_ds_sym, Vy_ds_sym, Ax_ds_sym, Ay_ds_sym]
    )
]
H, R, Vx, Vy, Ax, Ay = eval_sym([H_sym, R_sym, Vx_sym, Vy_sym, Ax_sym, Ay_sym], t, time)

# Mission Thrust, Power, Energy Profiles
Tx_sym , Ty_sym = get_thrust(H_sym, Vx_sym, Vy_sym, Ax_sym, Ay_sym, rho_0, M, g, Cd, Cl, Ab_frontal, Aw_planform, Ab_top)
T_sym, P_sym = get_power(Tx_sym, Ty_sym, H_sym, Vx_sym, Vy_sym, A_rotor, rotor_count, rho_0)
Tx, Ty, T, P = eval_sym([Tx_sym,Ty_sym,T_sym,P_sym], t, time)
E = cumulative_trapezoid(P, time, initial=0)

flight_angle_sym, tilt_angle_sym, tilt_speed_sym, tilt_acc_sym = get_rotor_tilt(Vx_sym, Vy_sym, Tx_sym, Ty_sym)
flight_angle, tilt_angle, tilt_speed, tilt_acc = eval_sym(
    [flight_angle_sym, tilt_angle_sym, tilt_speed_sym, tilt_acc_sym], t, time
)


# -----------------------------
# Flight Trajectory & Flight Path Angle
# -----------------------------
fig, axes = plt.subplots(6, 3, figsize=(14, 16))
phase_names = ["Climb", "Cruise", "Descent"]

H_phases = [H_cl, H_cr, H_ds]
R_phases = [R_cl, R_cr, R_ds]
Vx_phases = [Vx_cl, Vx_cr, Vx_ds]
Vy_phases = [Vy_cl, Vy_cr, Vy_ds]
Ax_phases = [Ax_cl, Ax_cr, Ax_ds]
Ay_phases = [Ay_cl, Ay_cr, Ay_ds]
t_phases = [t1, t2, t3]

for i in range(3):

    # Row 1: Trajectory
    axes[0, i].plot(R_phases[i]/1000, H_phases[i], lw=2)
    axes[0, i].set_title(f"{phase_names[i]} Trajectory")
    axes[0, i].set_ylabel("Altitude [m]")
    axes[0, i].set_xlabel("Range [km]")
    axes[0, i].grid(True)

    # Row 2: Flight path angle
    gamma = np.arctan2(np.abs(Vy_phases[i]), Vx_phases[i])
    if i == 0:
        gamma[0] = np.pi/2
    if i == 2:
        gamma = -gamma
        gamma[-1] = -np.pi/2

    axes[1, i].plot(t_phases[i]/60, np.degrees(gamma), lw=2)
    axes[1, i].set_ylabel("Flight Path γ [deg]")
    axes[1, i].grid(True)

    # Row 3: Vx
    axes[2, i].plot(t_phases[i]/60, Vx_phases[i], lw=2)
    axes[2, i].set_ylabel("Vx [m/s]")
    axes[2, i].grid(True)

    # Row 4: Vy
    axes[3, i].plot(t_phases[i]/60, Vy_phases[i], lw=2)
    axes[3, i].set_ylabel("Vy [m/s]")
    axes[3, i].grid(True)

    # Row 5: Ax
    axes[4, i].plot(t_phases[i]/60, Ax_phases[i], lw=2)
    axes[4, i].set_ylabel("Ax [m/s²]")
    axes[4, i].grid(True)

    # Row 6: Ay
    axes[5, i].plot(t_phases[i]/60, Ay_phases[i], lw=2)
    axes[5, i].set_ylabel("Ay [m/s²]")
    axes[5, i].set_xlabel("Time [min]")
    axes[5, i].grid(True)

plt.tight_layout()
plt.show()


# -----------------------------
# Plot Power & Energy
# -----------------------------
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()  # Create the twin axis once
fig.suptitle("Power & Energy Requirements", fontsize=16, fontweight='bold')

# Plot Power (Left Axis)
ln1 = ax1.plot(time, P/1000, color='tab:blue', linewidth=2, label="Power [kW]")
ax1.set_ylabel("Power [kW]", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot Energy & Max Energy (Right Axis)
ln2 = ax2.plot(time, E/3.6e6, color='tab:red', linewidth=2, linestyle='--', label="Energy [kWh]")
ln3 = ax2.plot(time, (E_max/3.6e6)*np.ones_like(time), color='red', linewidth=1.5, linestyle=':', label="70 % Capacity")
ln4 = ax2.plot(time, ((E_max/(1-reserve_energy_ratio))/3.6e6)*np.ones_like(time), color='red', linewidth=1.5, linestyle='-.', label="Total Capacity")
ax2.set_ylabel("Energy [kWh]", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Grid and Labels
ax1.set_xlabel("Time [s]")
ax1.grid(True, linestyle='--', alpha=0.4)

# Combined Legend
lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='best')

plt.tight_layout()
plt.show()
