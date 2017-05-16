"""Ant navigation along a single habitual route, controlled by Nengo."""

# Landmark parameters
lands_coll_name = "Landmarks"
land_radius = 1.2
normalize_lands_vecs = True
diff_alv_thres = 0.15
min_gain_diff_alv_dist = 0.6
max_gain_diff_alv_dist = 0.7

# Waypoint parameters
waypts_coll_name = "Waypoints"
init_waypt_dist_margin = 0.2  # m
final_waypt_dist_margin = 0.2  # m

# Robot parameters
robot_name = "Pioneer_p3dx"
motor_names = ["Pioneer_p3dx_leftMotor", "Pioneer_p3dx_rightMotor"]

# Nengo model parameters
sp_dim = 32
prb_syn = 0.01

# V-REP remote API parameters
vrep_ip = '127.0.0.1'
vrep_port = 19997

# Simulation parameters
sim_duration = 60.0  # s
sim_cycle_duration = 1.0  # s
nengo_sim_dt = 0.001  # s

# Other parameters
np_seed = 320364543
with_time = True
verbose = True
