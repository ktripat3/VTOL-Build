# -----------------------------
# Mission Parameters
# -----------------------------
gross_weight = 2404      # Gross Aircraft Weight (kg)
energy_capacity = 150    # Battery energy (kWh)
cruise_altitude = 4500   # Cruise altitude (m)
reserve_energy_ratio = 0.3
t_climb = 600
t_cruise = 2700
t_descent = 600

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
from tool_directory_num import get_flight_params, get_phases, solve_phase, print_performance_metrics, plot_time_trajectory, plot_range_trajectory, plot_energy_req

mission = [gross_weight, cruise_altitude, energy_capacity, reserve_energy_ratio]
aircraft = [wing_planform_area, aircraft_frontal_area, rotor_diameter, rotor_count, wing_drag_coefficient, wing_lift_coefficient]
phase_times = [t_climb, t_cruise, t_descent]

flight_params = get_flight_params(mission, aircraft)
phases, phase_params_dict = get_phases(flight_params, phase_times)

phase_results = {}
phase_performance = {}
for phase in phases:
    profile_params = phase_params_dict[phase]
    traj, performance = solve_phase(mission, aircraft, profile_params)
    
    phase_results[phase] = traj
    phase_performance[phase] = performance
    print_performance_metrics(performance)

    # plot_time_trajectory(phase_results)
    plot_range_trajectory(phase_results[phase], phase)
    plot_energy_req(mission, phase_results[phase])

print('Done')