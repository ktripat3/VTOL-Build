# -----------------------------
# Mission Parameters
# -----------------------------
gross_weight = 2404      # Gross Aircraft Weight (kg)
energy_capacity = 350    # Battery energy (kWh)
cruise_altitude = 4500   # Cruise altitude (m)
reserve_energy_ratio = 0.3
t_climb = 500
t_cruise = 500
t_descent = 500

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
from tool_directory_num import get_flight_params, get_phases, solve_phase, phase_time_optimizer, print_performance_metrics, plot_range_trajectory, plot_energy_req, plot_trajectory, plot_energy
import time

mission = [gross_weight, cruise_altitude, energy_capacity, reserve_energy_ratio]
aircraft = [wing_planform_area, aircraft_frontal_area, rotor_diameter, rotor_count, wing_drag_coefficient, wing_lift_coefficient]
phase_times = [t_climb, t_cruise, t_descent]

flight_params = get_flight_params(mission, aircraft)

start_time = time.perf_counter()
optimized_phase_times = phase_time_optimizer(phase_times, mission, flight_params)
end_time = time.perf_counter()
print(f"Optimization took {end_time - start_time:.4f} seconds")

phases, phase_params_dict = get_phases(flight_params, optimized_phase_times)
phase_results = {}
for phase in phases:
    profile_params = phase_params_dict[phase]
    traj, performance = solve_phase(flight_params, profile_params)
    phase_results[phase] = traj
    print_performance_metrics(performance, phase)

plot_trajectory(phase_results)
plot_energy(phase_results, mission)

print('Done')