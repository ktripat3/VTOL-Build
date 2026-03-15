# -----------------------------
# Mission Parameters
# -----------------------------
gross_weight = 2404      # Gross Aircraft Weight (kg)
energy_capacity = 150    # Battery energy (kWh)
cruise_altitude = 3048   # Cruise altitude (m)
reserve_energy_ratio = 0.4
t_climb = 240
t_cruise = 2700
t_descent = 300

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
unit_range = get_performance(controls)
plot_time_trajectory(controls) # Trajectory Plot
plot_range_trajectory(controls)
plot_energy_req(controls, mission)