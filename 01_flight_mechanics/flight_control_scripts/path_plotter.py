# -----------------------------
# Mission Parameters
# -----------------------------
gross_weight = 2404      # Gross Aircraft Weight (kg)
energy_capacity = 150    # Battery energy (kWh)
cruise_altitude = 4500   # Cruise altitude (m)
reserve_energy_ratio = 0.3
t_climb = 180
t_cruise = 2700
t_descent = 360

# -----------------------------
# Aircraft Parameters
# -----------------------------
rotor_count = 6
wing_planform_area = 12
aircraft_frontal_area = 3
rotor_diameter = 2.9
wing_drag_coefficient = 0.018
wing_lift_coefficient = 0.5


# -----------------------------
mission = [gross_weight, cruise_altitude, energy_capacity, t_climb, t_cruise, t_descent, reserve_energy_ratio]
aircraft = [wing_planform_area, aircraft_frontal_area, rotor_diameter, wing_drag_coefficient, wing_lift_coefficient, rotor_count]

from tool_directory import get_controls, get_performance, plot_time_trajectory, plot_range_trajectory, plot_energy_req
controls = get_controls(mission, aircraft)
unit_range, E_consumed, Range, T_rotor_max, P_rotor_max = get_performance(rotor_count, controls)

# ANSI color codes
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
print(f"{bcolors.OKGREEN}Total Range: {bcolors.END}{round(Range):,} km")
print(f"{bcolors.OKBLUE}Peak Thrust per Rotor: {bcolors.END}{round(T_rotor_max):,} kN")
print(f"{bcolors.OKCYAN}Peak Power per Rotor: {bcolors.END}{round(P_rotor_max):,} kW")
print(f"{bcolors.WARNING}Total Energy Consumption: {bcolors.END}{round(E_consumed):,} kWh")
print(f"{bcolors.BOLD}Range per Unit Energy Consumption: {bcolors.END}{round(unit_range,3):,} km/kWh")
print(bcolors.BOLD + bcolors.HEADER + "---------------------------------------------------------------\n" + bcolors.END)

# plot_time_trajectory(controls)
plot_range_trajectory(controls)
plot_energy_req(controls, mission)