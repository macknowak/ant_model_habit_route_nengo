"""Navigation using average landmark vector.

This script provides a model that uses the average landmark vector method to
navigate a robot to the goal. During navigation the average landmark vector
computed based on the current robot pose is compared with the average landmark
vector computed at the goal location for the desired robot orientation at the
goal, and the resulting difference is used to determine the robot heading.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import math

import numpy as np
import vrepsim as vrs

# --- PARAMETERS ---
class Params(object): pass
params = Params()

# Goal parameters
params.goal_name = "Goal"
params.goal_dist_margin = 0.2  # m

# Landmark parameters
params.land1_name = "Landmark1"
params.land2_name = "Landmark2"
params.land3_name = "Landmark3"

# Robot parameters
params.robot_name = "Pioneer_p3dx"
params.motor_names = ["Pioneer_p3dx_leftMotor", "Pioneer_p3dx_rightMotor"]
params.robot_goal_orient_deg = 180  # deg

# V-REP remote API parameters
params.vrep_ip = '127.0.0.1'
params.vrep_port = 19997

# Simulation parameters
params.sim_duration = 10.0  # s

# Other parameters
params.verbose = True
# ------------------

# Convert robot orientation at the goal to radians
robot_goal_orient = params.robot_goal_orient_deg * math.pi / 180.0

# Connect to V-REP
vrep_sim = vrs.Simulator(params.vrep_ip, params.vrep_port)
vrep_sim.connect(verbose=params.verbose)

try:
    # Retrieve V-REP simulation time step
    vrep_sim_dt = vrep_sim.get_sim_dt()

    # Create representation of the robot
    bot = vrs.PioneerBot(vrep_sim, params.robot_name, None, params.motor_names)

    # Create representation of the goal
    goal = vrs.Dummy(vrep_sim, params.goal_name)

    # Create representation of the landmarks
    land1 = vrs.Dummy(vrep_sim, params.land1_name)
    land2 = vrs.Dummy(vrep_sim, params.land2_name)
    land3 = vrs.Dummy(vrep_sim, params.land3_name)

    # Retrieve goal position
    goal_pos = goal.get_position()[:2]

    # Retrieve landmark positions relative to the goal and transform them to
    # account for the robot orientation at the goal
    c, s = math.cos(-robot_goal_orient), math.sin(-robot_goal_orient)
    land_rot = np.array([[c, -s], [s, c]])
    land1_loc_pos_goal = np.dot(land_rot, land1.get_position(goal.handle)[:2])
    land2_loc_pos_goal = np.dot(land_rot, land2.get_position(goal.handle)[:2])
    land3_loc_pos_goal = np.dot(land_rot, land3.get_position(goal.handle)[:2])

    # Determine average landmark vector at the goal
    avg_land_vect_goal = (land1_loc_pos_goal + land2_loc_pos_goal
                          + land3_loc_pos_goal)

    # Start V-REP simulation in synchronous operation mode
    vrep_sim.start_sim(params.verbose)

    try:
        # Pursue the goal based on landmark positions
        sim_time = 0.0
        while sim_time < params.sim_duration:
            # Trigger next V-REP simulation step
            vrep_sim.trig_sim_step()
            sim_time += vrep_sim_dt

            # Retrieve robot position
            bot_pos = bot.get_position()[:2]

            # Determine distance to the goal
            goal_distance = math.sqrt(
                (goal_pos[0] - bot_pos[0])**2 + (goal_pos[1] - bot_pos[1])**2)

            # If the goal has been reached, stop the wheels and stop moving
            if goal_distance <= params.goal_dist_margin:
                bot.wheels.set_velocities((0.0, 0.0))
                if params.verbose:
                    print("Goal reached.")
                    print("Simulation time: {:.3f} s.".format(sim_time))
                    print("Distance to goal: {:.2f} m.".format(goal_distance))
                break

            # Retrieve landmark positions relative to the robot
            land1_loc_pos = np.array(land1.get_position(bot.handle)[:2])
            land2_loc_pos = np.array(land2.get_position(bot.handle)[:2])
            land3_loc_pos = np.array(land3.get_position(bot.handle)[:2])

            # Determine average landmark vector
            avg_land_vect = land1_loc_pos + land2_loc_pos + land3_loc_pos

            # Update wheel speeds
            def calc_wheel_speed(x):
                if np.sqrt(x[0]**2 + x[1]**2) < 0.2:
                    return [0.0, 0.0]
                theta = np.arctan2(x[1], x[0])
                if -0.3 < theta < 0.3:
                    return [5.0, 5.0]
                else:
                    return [-5.0*theta, 5.0*theta]

            bot.wheels.set_velocities(
                calc_wheel_speed(avg_land_vect - avg_land_vect_goal))

        else:
            # If the goal has not been reached until the time is up, stop the
            # wheels and stop moving
            bot.wheels.set_velocities((0.0, 0.0))
            if params.verbose:
                print("Time up, goal not reached.")
                print("Distance to goal: {:.2f} m.".format(goal_distance))

        if params.verbose:
            print("Goal AVL: " + str(avg_land_vect_goal))
            print("Robot AVL: " + str(avg_land_vect))

    finally:
        # Stop V-REP simulation
        vrep_sim.stop_sim(params.verbose)

finally:
    # Disconnect from V-REP
    vrep_sim.disconnect(params.verbose)
