"""Navigation using average landmark vector, controlled by Nengo.

This script provides a Nengo model that uses the average landmark vector method
to navigate a robot to the goal. During navigation the average landmark vector
computed based on the current robot pose is compared with the average landmark
vector computed at the goal location for the desired robot orientation at the
goal, and the resulting difference is used to determine the robot heading.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo
import numpy as np

with_vrep = False
if with_vrep:
    import vrepsim as vrs
else:
    bot_init_pos = [1.0, 0.0]
    bot_init_orient_deg = -180
    goal_pos = [-0.5, 0.0]
    land1_pos = [-0.5, -0.5]
    land2_pos = [-1.0, 0.0]
    land3_pos = [-0.5, 0.5]

# --- PARAMETERS ---
class Params(object): pass
params = Params()

if with_vrep:
    # Goal parameters
    params.goal_name = "Goal"

    # Landmark parameters
    params.land1_name = "Landmark1"
    params.land2_name = "Landmark2"
    params.land3_name = "Landmark3"

    # Robot parameters
    params.robot_name = "Pioneer_p3dx"
    params.motor_names = ["Pioneer_p3dx_leftMotor", "Pioneer_p3dx_rightMotor"]

    # Remote API parameters
    params.vrep_ip = '127.0.0.1'
    params.vrep_port = 19997

    # Simulation parameters
    params.nengo_sim_dt = 0.001  # s

    # Other parameters
    params.verbose = True

# Robot parameters
params.bot_init_goal_orient_deg = -180

# Nengo model parameters
params.model_seed = None
# ------------------

# Connect to V-REP
if with_vrep:
    vrep_sim = vrs.Simulator(params.vrep_ip, params.vrep_port)
    vrep_sim.connect(verbose=params.verbose)

try:
    # Validate V-REP simulation time step
    if with_vrep:
        vrep_sim_dt = vrep_sim.get_sim_dt()
        sim_dt_ratio = vrep_sim_dt / params.nengo_sim_dt
        assert sim_dt_ratio == round(sim_dt_ratio), \
            ("V-REP simulation time step must be evenly divisible by Nengo "
             "simulation time step.")

    # Create representation of the robot
    if with_vrep:
        bot = vrs.PioneerBot(vrep_sim, params.robot_name, None,
                             params.motor_names)

    # Create representation of the goal
    if with_vrep:
        goal = vrs.Dummy(vrep_sim, params.goal_name)

    # Create representation of the landmarks
    if with_vrep:
        land1 = vrs.Dummy(vrep_sim, params.land1_name)
        land2 = vrs.Dummy(vrep_sim, params.land2_name)
        land3 = vrs.Dummy(vrep_sim, params.land3_name)

    # Create communicator for data exchange with V-REP
    if with_vrep:
        vrep_comm = vrs.nengo.NengoComm(vrep_sim, sim_dt_ratio)

    # Create Nengo model controlling the robot
    model = nengo.Network(seed=params.model_seed)
    with model:
        # Create node representing communicator for data exchange with V-REP
        if with_vrep:
            vrep_proxy = nengo.Node(vrep_comm, size_in=2, size_out=11)

        # Create node representing robot position
        if with_vrep:
            bot_pos_input = nengo.Node(None, size_in=2)
            vrep_comm.add_output(lambda: bot.get_position()[:2], 2)
            nengo.Connection(vrep_proxy[0:2], bot_pos_input, synapse=None)
        else:
            bot_pos_input = nengo.Node(bot_init_pos)

        # Create node representing robot orientation
        if with_vrep:
            bot_orient_input = nengo.Node(None, size_in=1)
            vrep_comm.add_output(lambda: bot.get_orientation()[2], 1)
            nengo.Connection(vrep_proxy[2], bot_orient_input, synapse=None)
        else:
            bot_orient_input = nengo.Node(bot_init_orient_deg * np.pi / 180.0)

        # Create node representing goal position
        if with_vrep:
            goal_pos_input = nengo.Node(None, size_in=2)
            vrep_comm.add_output(lambda: goal.get_position()[:2], 2)
            nengo.Connection(vrep_proxy[3:5], goal_pos_input, synapse=None)
        else:
            goal_pos_input = nengo.Node(goal_pos)

        # Create node representing robot orientation at the goal
        bot_goal_orient_input = nengo.Node(params.bot_init_goal_orient_deg)

        # Create node representing position of the first landmark
        if with_vrep:
            land1_pos_input = nengo.Node(size_in=2)
            vrep_comm.add_output(lambda: land1.get_position()[:2], 2)
            nengo.Connection(vrep_proxy[5:7], land1_pos_input, synapse=None)
        else:
            land1_pos_input = nengo.Node(land1_pos)

        # Create node representing position of the second landmark
        if with_vrep:
            land2_pos_input = nengo.Node(size_in=2)
            vrep_comm.add_output(lambda: land2.get_position()[:2], 2)
            nengo.Connection(vrep_proxy[7:9], land2_pos_input, synapse=None)
        else:
            land2_pos_input = nengo.Node(land2_pos)

        # Create node representing position of the third landmark
        if with_vrep:
            land3_pos_input = nengo.Node(size_in=2)
            vrep_comm.add_output(lambda: land3.get_position()[:2], 2)
            nengo.Connection(vrep_proxy[9:11], land3_pos_input, synapse=None)
        else:
            land3_pos_input = nengo.Node(land3_pos)

        # Create node representing average landmark vector at the goal
        def calc_alv_goal(t, x):
            bot_orient_goal = x[2] * np.pi / 180.0
            c, s = np.cos(-bot_orient_goal), np.sin(-bot_orient_goal)
            land_rot = np.array([[c, -s], [s, c]])
            land1_loc_pos_goal = np.array([x[3]-x[0], x[4]-x[1]])
            land1_loc_pos_goal = np.dot(land_rot, land1_loc_pos_goal)
            land2_loc_pos_goal = np.array([x[5]-x[0], x[6]-x[1]])
            land2_loc_pos_goal = np.dot(land_rot, land2_loc_pos_goal)
            land3_loc_pos_goal = np.array([x[7]-x[0], x[8]-x[1]])
            land3_loc_pos_goal = np.dot(land_rot, land3_loc_pos_goal)
            return land1_loc_pos_goal + land2_loc_pos_goal + land3_loc_pos_goal

        avg_land_vect_goal = nengo.Node(calc_alv_goal, size_in=9, size_out=2)
        nengo.Connection(goal_pos_input, avg_land_vect_goal[0:2], synapse=None)
        nengo.Connection(bot_goal_orient_input, avg_land_vect_goal[2],
                         synapse=None)
        nengo.Connection(land1_pos_input, avg_land_vect_goal[3:5],
                         synapse=None)
        nengo.Connection(land2_pos_input, avg_land_vect_goal[5:7],
                         synapse=None)
        nengo.Connection(land3_pos_input, avg_land_vect_goal[7:9],
                         synapse=None)

        # Create node representing average landmark vector relative to the
        # robot
        def calc_alv_bot(t, x):
            c, s = np.cos(-x[2]), np.sin(-x[2])
            land_rot = np.array([[c, -s], [s, c]])
            land1_loc_pos = np.array([x[3]-x[0], x[4]-x[1]])
            land1_loc_pos = np.dot(land_rot, land1_loc_pos)
            land2_loc_pos = np.array([x[5]-x[0], x[6]-x[1]])
            land2_loc_pos = np.dot(land_rot, land2_loc_pos)
            land3_loc_pos = np.array([x[7]-x[0], x[8]-x[1]])
            land3_loc_pos = np.dot(land_rot, land3_loc_pos)
            return land1_loc_pos + land2_loc_pos + land3_loc_pos

        avg_land_vect = nengo.Node(calc_alv_bot, size_in=9, size_out=2)
        nengo.Connection(bot_pos_input, avg_land_vect[0:2], synapse=None)
        nengo.Connection(bot_orient_input, avg_land_vect[2], synapse=None)
        nengo.Connection(land1_pos_input, avg_land_vect[3:5], synapse=None)
        nengo.Connection(land2_pos_input, avg_land_vect[5:7], synapse=None)
        nengo.Connection(land3_pos_input, avg_land_vect[7:9], synapse=None)

        # Create ensemble representing difference between the average landmark
        # vector relative to the robot and the average landmark vector at the
        # goal
        diff_avg_land_vect = nengo.Ensemble(100, dimensions=2, radius=7)
        nengo.Connection(avg_land_vect, diff_avg_land_vect, transform=1)
        nengo.Connection(avg_land_vect_goal, diff_avg_land_vect, transform=-1)

        # Create ensemble representing wheel speeds
        def calc_wheel_speeds(x):
            if np.sqrt(x[0]**2 + x[1]**2) < 0.2:
                return [0.0, 0.0]
            theta = np.arctan2(x[1], x[0])
            if -0.3 < theta < 0.3:
                return [5.0, 5.0]
            else:
                return [-5.0*theta, 5.0*theta]

        wheel_speeds = nengo.Ensemble(100, dimensions=2, radius=1.0)
        nengo.Connection(diff_avg_land_vect, wheel_speeds,
                         function=calc_wheel_speeds)
        if with_vrep:
            vrep_comm.add_input(bot.wheels.set_velocities, 2)
            nengo.Connection(wheel_speeds, vrep_proxy[0:2])

    if with_vrep:
        # Start V-REP simulation in synchronous operation mode
        vrep_sim.start_sim(params.verbose)

except Exception:
    if with_vrep:
        vrep_sim.disconnect(params.verbose)
    raise
