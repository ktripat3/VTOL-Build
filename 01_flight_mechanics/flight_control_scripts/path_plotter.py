import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tool_directory import solve_phase

# -----------------------------
# Constants
# -----------------------------
rho_0 = 1.225               # Air Density @ Sea Level (kg/m^3)
g = 9.81                    # Acceleration due to gravity (m/s^2)
reserve_energy_ratio = 0.4  # Ratio of reserve energy to total energy


# -----------------------------
# Mission Parameters
# -----------------------------
gross_weight = 2404     # Gross Aircraft Weight (kg)
cruise_speed = 322      # Cruise speed (km/h)
cruise_altitude = 3048   # Cruise altitude (m)
energy_capacity = 150    # Battery energy (kWh)
t_climb = 420
t_cruise = 1800


# -----------------------------
# Aircraft Parameters
# -----------------------------
wing_drag_coefficient = 0.018
aircraft_frontal_area = 3
rotor_diameter = 2.9
rotor_count = 6
wing_planform_area = 12


# -----------------------------
M = gross_weight
A = cruise_altitude
Vc = (5/18) * cruise_speed
Aw_planform = wing_planform_area
Ab_frontal = aircraft_frontal_area
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
time = np.concatenate([t1, t2, t3])


# -----------------------------
# Compute Flight Controls
# -----------------------------
# Climb Profiles
cl = solve_phase(
    0, A,
    0,
    0, Vc,
    t_climb,
    0,
    t1,
    rho_0, M, g, Cd, Cl, Ab_frontal, Aw_planform, A_rotor, rotor_count
)

# Cruise Profiles
cr = solve_phase(
    A, A,
    cl["R"][-1],
    Vc, Vc,
    t_cruise,
    t_climb,
    t2,
    rho_0, M, g, Cd, Cl, Ab_frontal, Aw_planform, A_rotor, rotor_count
)

# Descent Profiles
ds = solve_phase(
    A, 0,
    cr["R"][-1],
    Vc, 0,
    t_climb,
    t_climb + t_cruise,
    t3,
    rho_0, M, g, Cd, Cl, Ab_frontal, Aw_planform, A_rotor, rotor_count
)

(H_cl,R_cl,Vx_cl,Vy_cl,Ax_cl,Ay_cl,Tx_cl,Ty_cl, T_cl, P_cl,
 flight_angle_cl,tilt_angle_cl,tilt_speed_cl,tilt_acc_cl) = cl.values()

(H_cr,R_cr,Vx_cr,Vy_cr,Ax_cr,Ay_cr,Tx_cr,Ty_cr, T_cr, P_cr,
 flight_angle_cr,tilt_angle_cr,tilt_speed_cr,tilt_acc_cr) = cr.values()

(H_ds,R_ds,Vx_ds,Vy_ds,Ax_ds,Ay_ds,Tx_ds,Ty_ds, T_ds, P_ds,
 flight_angle_ds,tilt_angle_ds,tilt_speed_ds,tilt_acc_ds) = ds.values()


# -----------------------------
# Trajectory Plot
# -----------------------------
fig, axes = plt.subplots(6, 3, figsize=(15,12))
flight = {
    "Climb": cl,
    "Cruise": cr,
    "Descent": ds
}

t_phases = [t1, t2, t3]
phase_names = list(flight.keys())
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

flight_angle_phases = [p["flight_angle"] for p in phases]
flight_angle_phases[-1][-1] = 0

tilt_angle_phases = [p["tilt_angle"] for p in phases]
tilt_speed_phases = [p["tilt_speed"] for p in phases]
tilt_acc_phases = [p["tilt_acc"] for p in phases]


# Grayscale palette
primary_color   = '#d9885f'  # dull/muted orange
secondary_color = '#7fa97f'  # dull/muted green
tertiary_color  = '#dcdcdc'  # light gray for grids

for i in range(3):

    # 1. Ground Range & Altitude
    ax1 = axes[0,i]
    ax1.plot(t_phases[i], R_phases[i]/1000, color=primary_color, linestyle='-', label='Ground Range')
    ax1.set_title(phase_names[i], fontsize=12, fontweight='bold')
    ax1.set_ylabel("Range [km]", color=primary_color)
    ax1.tick_params(axis='y', labelcolor=primary_color)
    ax1.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    ax1b = ax1.twinx()
    ax1b.plot(t_phases[i], H_phases[i], color=secondary_color, linestyle='--', label='Altitude')
    ax1b.set_ylabel("Altitude [m]", color=secondary_color)
    ax1b.tick_params(axis='y', labelcolor=secondary_color)

    # 2. Flight Path Angle & Tilt Angle
    ax2 = axes[1,i]
    ax2.plot(t_phases[i], (180/np.pi)*flight_angle_phases[i], color=primary_color, linestyle='-', label='Flight Path')
    ax2.set_ylabel("Flight Path [deg]", color=primary_color)
    ax2.tick_params(axis='y', labelcolor=primary_color)
    ax2.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    ax2b = ax2.twinx()
    ax2b.plot(t_phases[i], (180/np.pi)*tilt_angle_phases[i], color=secondary_color, linestyle='--', label='Tilt Angle')
    ax2b.set_ylabel("Tilt [deg]", color=secondary_color)
    ax2b.tick_params(axis='y', labelcolor=secondary_color)

    # 3. Vx & Vy
    ax3 = axes[2,i]
    ax3.plot(t_phases[i], Vx_phases[i], color=primary_color, linestyle='-', label='Vx')
    ax3.set_ylabel("Vx [m/s]", color=primary_color)
    ax3.tick_params(axis='y', labelcolor=primary_color)
    ax3.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    ax3b = ax3.twinx()
    ax3b.plot(t_phases[i], Vy_phases[i], color=secondary_color, linestyle='--', label='Vy')
    ax3b.set_ylabel("Vy [m/s]", color=secondary_color)
    ax3b.tick_params(axis='y', labelcolor=secondary_color)

    # 4. Tilt Speed
    axes[3,i].plot(t_phases[i], (180/np.pi)*tilt_speed_phases[i], color=primary_color, linestyle='-')
    axes[3,i].set_ylabel("Tilt Speed [deg/s]")
    axes[3,i].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    # 5. Ax & Ay
    ax5 = axes[4,i]
    ax5.plot(t_phases[i], Ax_phases[i], color=primary_color, linestyle='-', label='Ax')
    ax5.set_ylabel("Ax [m/s²]", color=primary_color)
    ax5.tick_params(axis='y', labelcolor=primary_color)
    ax5.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

    ax5b = ax5.twinx()
    ax5b.plot(t_phases[i], Ay_phases[i], color=secondary_color, linestyle='--', label='Ay')
    ax5b.set_ylabel("Ay [m/s²]", color=secondary_color)
    ax5b.tick_params(axis='y', labelcolor=secondary_color)

    # 6. Tilt Acceleration
    axes[5,i].plot(t_phases[i], (180/np.pi)*tilt_acc_phases[i], color=primary_color, linestyle='-')
    axes[5,i].set_ylabel("Tilt Acc [deg/s²]")
    axes[5,i].set_xlabel("Time [s]")
    axes[5,i].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()


# -----------------------------
# Compute total thrust and power/energy
# -----------------------------
T = np.concatenate(T_phases)
P = np.concatenate(P_phases)
E = np.array([np.trapz(P[:i+1], time[:i+1]) for i in range(len(time))])

fig, axes = plt.subplots(1, 2, figsize=(15,5))
fig.suptitle("Thrust, Power & Energy", fontsize=16, fontweight='bold')

# -----------------------------
# Total Thrust
# -----------------------------
axes[0].plot(time, T/1000, color=primary_color, linewidth=2, label='Total Thrust')
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Thrust [kN]")
axes[0].grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)
axes[0].legend(loc='best')

# -----------------------------
# Power (Left) and Energy (Right)
# -----------------------------
ax1 = axes[1]
ax2 = ax1.twinx()

# Power [kW]
ln1 = ax1.plot(time, P/1000, color='tab:blue', linewidth=2, label="Power [kW]")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Power [kW]", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, color=tertiary_color, linestyle='--', linewidth=0.5)

# Energy [kWh]
ln2 = ax2.plot(time, E/3.6e6, color='tab:red', linewidth=2, linestyle='-', label="Energy [kWh]")
ln3 = ax2.plot(time, (E_max/3.6e6)*np.ones_like(time), color='red', linewidth=1.5, linestyle=':', label="70 % Capacity")
ln4 = ax2.plot(time, ((E_max/(1-reserve_energy_ratio))/3.6e6)*np.ones_like(time), color='red', linewidth=1.5, linestyle='-.', label="Total Capacity")

ax2.set_ylabel("Energy [kWh]", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Combined legend
lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='best')

plt.tight_layout()
plt.show()
