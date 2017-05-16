"""Ant navigation along a single habitual route, controlled by Nengo.

This script provides a Nengo model that controls a differential drive robot
simulated using V-REP simulator, which mimics a desert ant approaching the
nest along a habitual route. Communication between the Nengo model and V-REP is
handled using the V-REP remote API.

The robot is supposed to move from the start waypoint to the intermediate
waypoint and then to the final waypoint. Arrival at each waypoint is determined
by comparing the average landmark vector that is currently perceived with the
average landmark vector corresponding to that perceived at the waypoint, and
then comparing the resulting difference against a threshold. The direction in
which the robot should move is a combination of 2 factors: (1) the normalized
(local) vector associated with the previous waypoint, which points toward the
next waypoint; and (2) the difference between the average landmark vector (that
is currently perceived) and the average landmark vector corresponding to the
next waypoint. The contributions of these factors are dynamically weighted
using gains that change depending on how far from the previous waypoint and
thus how close to the next waypoint the robot is.

The Nengo model comprises the following core objects:

- a node representing robot position (obtained from V-REP);
- a node representing robot orientation (obtained from V-REP);
- a node representing average landmark vector (calculated directly);
- an ensemble representing lack of average landmark vector (to indicate whether
  the value representing average landmark vector is meaningful, which is not
  the case when there are no landmarks in view);
- a state representing previous waypoint (as a semantic pointer);
- an associative memory between previous waypoints and normalized vectors to
  the next waypoints;
- a state representing normalized vector to the next waypoint;
- an ensemble representing normalized vector to the next waypoint relative to
  the robot;
- an associative memory between previous and next waypoints;
- a state representing next waypoint (as a semantic pointer);
- an associative memory between waypoints and corresponding average landmark
  vectors;
- a state representing average landmark vector at the next waypoint;
- a node representing normalized distance from the previous waypoint;
- an ensemble representing gain of the difference between the average landmark
  vector and the average landmark vector at the next waypoint;
- an ensemble representing difference between the average landmark vector and
  the average landmark vector at the next waypoint;
- an ensemble representing thresholded magnitude of the difference between the
  average landmark vector and the average landmark vector at the next waypoint;
- an ensemble representing velocity vector;
- an ensemble representing wheel speeds (sent to V-REP);
- an action selection circuit.

The robot model used in V-REP is Pioneer_p3dx. It has several sensors, 2
motor-driven wheels (1 on the left and 1 on the right), and 1 caster wheel (in
the back). None of the sensors are used because robot position as well as
positions of landmarks are obtained directly. Each motor-driven wheel receives
signals controlling its velocity from the Nengo model.

To successfully run this script, V-REP simulator must first be launched with a
continuous remote API server service started and then scene file 'scene.ttt'
must be opened in the simulator. To allow this script to remotely control the
Pioneer_p3dx robot that is part of this scene, the original child script
associated with this robot was removed from the scene file.

Moreover, to successfully run this script, the following V-REP files (or,
alternatively, links to them) have to exist in the current directory so that
the V-REP remote API could be used (here, 'VREP_DIR' denotes the directory in
which V-REP is installed):

- 'vrep.py' (original file in: 'VREP_DIR/programming/remoteApiBindings/python/
  python/');
- 'vrepConst.py' (original file in: 'VREP_DIR/programming/remoteApiBindings/
  python/python/');
- ['remoteApi.dll' | 'remoteApi.dylib' | 'remoteApi.so'] (original file in:
  'VREP_DIR/programming/remoteApiBindings/lib/lib/[32Bit | 64Bit]/').

This script was developed for Nengo 2.4.0 and V-REP 3.4.0.
"""

__version__ = '1.0'
__author__ = "Przemyslaw (Mack) Nowak, Terry Stewart"

import os
import platform
import sys

import nengo
import nengo.spa as spa
import numpy as np
import vrepsim as vrs

# Filenames
bot_pos_filename = "botpos.txt"
bot_orients_filename = "botorient.txt"
bot_alvs_filename = "botalv.txt"
waypts_pos_filename = "wayptpos.txt"
waypts_alvs_filename = "wayptalv.txt"
waypts_names_filename = "wayptname.txt"
prev_waypts_filename = "prevwaypt.txt"
prev_waypt_neurons_filename = "prevwayptnrn.npz"
next_waypts_filename = "nextwaypt.txt"
next_waypt_neurons_filename = "nextwayptnrn.npz"
lands_pos_filename = "landpos.txt"
diffs_alv_filename = "diffalv.txt"
gains_diff_alv_filename = "gaindiffalv.txt"
bg_filename = "bg.txt"
thal_filename = "thal.txt"
thal_neurons_filename = "thalnrn.npz"
params_filename = "param.yml"
versions_filename = "version.yml"

# Parameters to be saved
saved_params = [
    'land_radius', 'normalize_lands_vecs', 'diff_alv_thres',
    'min_gain_diff_alv_dist', 'max_gain_diff_alv_dist',
    'init_waypt_dist_margin', 'final_waypt_dist_margin', 'sp_dim', 'prb_syn',
    'sim_duration', 'sim_cycle_duration', 'nengo_sim_dt', 'vrep_sim_dt',
    'np_seed'
    ]

# Process command line arguments
n_args = len(sys.argv) - 1
if n_args < 1:
    sys.stderr.write("{}: error: too few arguments\n".format(sys.argv[0]))
    sys.exit(2)
if n_args > 2:
    sys.stderr.write("{}: error: too many arguments\n".format(sys.argv[0]))
    sys.exit(2)
if n_args > 1:
    data_dirname = sys.argv[2]
    save_data = True
else:
    save_data = False

# Load parameters
class Params(dict): pass
params = Params()
with open(sys.argv[1]) as params_file:
    exec(params_file.read(), globals(), params)
for paramname, paramval in params.items():
    setattr(params, paramname, paramval)

# If necessary, create data directory
if save_data:
    os.mkdir(data_dirname)

# Set random seed
np.random.seed(params.np_seed)

# Connect to V-REP
vrep_sim = vrs.Simulator(params.vrep_ip, params.vrep_port)
vrep_sim.connect(verbose=params.verbose)

try:
    # Validate distances for the minimum and maximum gains of differences
    # between average landmark vectors
    assert params.min_gain_diff_alv_dist <= params.max_gain_diff_alv_dist, \
        ("Distance for the minimum gain of differences between average "
         "landmark vectors must not be greater than distance for the maximum "
         "gain.")

    # Determine difference between distances for the minimum and maximum gains
    # of differences between average landmark vectors
    diff_gain_diff_alv_dist = (params.max_gain_diff_alv_dist
                               - params.min_gain_diff_alv_dist)

    # Validate simulation duration
    n_sim_cycles = params.sim_duration / params.sim_cycle_duration
    assert n_sim_cycles == round(n_sim_cycles), \
        ("Simulation duration must be evenly divisible by simulation cycle "
         "duration.")

    # Validate V-REP simulation time step
    params.vrep_sim_dt = params['vrep_sim_dt'] = vrep_sim.get_sim_dt()
    sim_dt_ratio = params.vrep_sim_dt / params.nengo_sim_dt
    assert sim_dt_ratio == round(sim_dt_ratio), \
        ("V-REP simulation time step must be evenly divisible by Nengo "
         "simulation time step.")

    # Create communicator for data exchange with V-REP
    sim_dt_ratio = int(sim_dt_ratio)
    vrep_comm = vrs.nengo.NengoComm(vrep_sim, sim_dt_ratio)

    # Create representation of the robot
    bot = vrs.PioneerBot(vrep_sim, params.robot_name, None, params.motor_names)

    # Retrieve positions of waypoints
    waypts_coll = vrs.Collection(vrep_sim, params.waypts_coll_name)
    waypts_pos = np.array(waypts_coll.get_positions(), dtype=float)[:,:2]

    # Validate number of waypoints
    n_waypts = len(waypts_pos)
    assert n_waypts >= 2, "Number of waypoints must not be less than 2."

    # Validate robot initial position
    bot_pos = np.array(bot.get_position()[:2], dtype=float)
    bot_waypt_dist = np.sqrt(((waypts_pos[0] - bot_pos)**2).sum())
    assert bot_waypt_dist < params.init_waypt_dist_margin, \
        ("Robot initial position must be closer than {:.2f} m from the first "
         "waypoint.".format(params.init_waypt_dist_margin))

    # Retrieve names of waypoints
    waypts_names = waypts_coll.get_names()

    # Retrieve positions of landmarks
    lands_coll = vrs.Collection(vrep_sim, params.lands_coll_name)
    lands_pos = np.array(lands_coll.get_positions(), dtype=float)[:,:2]

    # Determine vectors between waypoints and normalized versions of these
    # vectors
    waypts_vecs = np.zeros((n_waypts,2))
    waypts_vecs[:-1] = np.diff(waypts_pos, axis=0)
    waypts_vecs_norms = np.sqrt((waypts_vecs**2).sum(axis=1, keepdims=True))
    norm_waypts_vecs = np.zeros((n_waypts,2))
    norm_waypts_vecs[:-1] = waypts_vecs[:-1] / waypts_vecs_norms[:-1]

    # Determine distances between waypoints
    waypts_dists = np.zeros(n_waypts)
    waypts_dists[:-1] = np.sqrt((waypts_vecs[:-1]**2).sum(axis=1))

    # Determine average landmark vectors at waypoints except for the initial
    # waypoint
    waypts_alvs = np.zeros((n_waypts,2))
    for w in range(1, n_waypts):
        lands_trans_pos = lands_pos - waypts_pos[w]
        lands_loc_dists = np.sqrt((lands_trans_pos**2).sum(axis=1))
        waypt_lands_trans_pos = \
            lands_trans_pos[lands_loc_dists<=params.land_radius]
        if params.normalize_lands_vecs:
            waypt_lands_trans_pos /= np.sqrt(
                (waypt_lands_trans_pos**2).sum(axis=1, keepdims=True))
        x, y = norm_waypts_vecs[w-1]
        land_rot = np.array([[x, y], [-y, x]], dtype=float)
        waypt_lands_loc_pos = np.dot(land_rot, waypt_lands_trans_pos.T).T
        waypts_alvs[w] = (waypt_lands_loc_pos.sum(axis=0)
                          / len(waypt_lands_loc_pos))

    # Validate average landmark vectors at waypoints
    assert not np.isnan(waypts_alvs).any(), \
        ("Average landmark vector at waypoint {} could not be determined."
         "".format(waypts_names[np.where(np.isnan(waypts_alvs[:,0]))[0][0]]))

    # Create vocabularies
    v_waypts = spa.Vocabulary(params.sp_dim)
    v_waypts.extend(waypts_names)
    v_norm_waypts_vecs = spa.Vocabulary(2)
    for w, norm_waypt_vec in enumerate(norm_waypts_vecs):
        v_norm_waypts_vecs.add(waypts_names[w], norm_waypt_vec)
    v_waypts_alvs = spa.Vocabulary(2)
    for w, waypt_alv in enumerate(waypts_alvs):
        v_waypts_alvs.add(waypts_names[w], waypt_alv)
    v_thres_diff_alv = spa.Vocabulary(1)
    v_thres_diff_alv.add('REACHED', [1.0])

    # Create Nengo model controlling the robot
    model = spa.SPA()
    with model:
        # Create node representing communicator for data exchange with V-REP
        vrep_proxy = nengo.Node(vrep_comm, size_in=2, size_out=3)

        # Create node representing robot position
        bot_pos_inp = nengo.Node(None, size_in=2)
        vrep_comm.add_output(lambda: bot.get_position()[:2], 2)
        nengo.Connection(vrep_proxy[0:2], bot_pos_inp, synapse=None)

        # Create node representing robot orientation
        bot_orient_inp = nengo.Node(lambda t, x: x, size_in=1, size_out=1)
        vrep_comm.add_output(lambda: bot.get_orientation()[2], 1)
        nengo.Connection(vrep_proxy[2], bot_orient_inp, synapse=None)

        # Create node representing average landmark vector
        def calc_bot_alv(t, x):
            lands_trans_pos = lands_pos - x[0:2]
            lands_loc_dists = np.sqrt((lands_trans_pos**2).sum(axis=1))
            bot_lands_trans_pos = \
                lands_trans_pos[lands_loc_dists<=params.land_radius]
            if len(bot_lands_trans_pos):
                if params.normalize_lands_vecs:
                    bot_lands_trans_pos /= np.sqrt(
                        (bot_lands_trans_pos**2).sum(axis=1, keepdims=True))
                y = np.ones(3, dtype=float)
                c, s = np.cos(-x[2]), np.sin(-x[2])
                land_rot = np.array([[c, -s], [s, c]], dtype=float)
                bot_lands_loc_pos = np.dot(land_rot, bot_lands_trans_pos.T).T
                y[0:2] = bot_lands_loc_pos.sum(axis=0) / len(bot_lands_loc_pos)
                return y
            else:
                return np.zeros(3)

        bot_alv_inp = nengo.Node(calc_bot_alv, size_in=3, size_out=3)
        nengo.Connection(bot_pos_inp, bot_alv_inp[0:2], synapse=None)
        nengo.Connection(bot_orient_inp, bot_alv_inp[2], synapse=None)

        # Create ensemble representing lack of average landmark vector
        bot_alv_off = nengo.Ensemble(
            50, dimensions=1, encoders=nengo.dists.Choice([[1.0]]),
            intercepts=nengo.dists.Uniform(0.5, 1.0))
        nengo.Connection(bot_alv_inp[2], bot_alv_off,
                         function=lambda x: 1.0 - x)

        # Create state representing previous waypoint
        model.prev_waypt = spa.State(params.sp_dim, vocab=v_waypts, feedback=1)

        # Create input initializing previous waypoint
        model.init_prev_waypt = spa.Input(
            prev_waypt=lambda t: waypts_names[0] if t < 0.05 else '0')

        # Create associative memory between previous waypoints and normalized
        # vectors to the next waypoints
        model.waypts_prev2vec = spa.AssociativeMemory(
            input_vocab=v_waypts,
            output_vocab=v_norm_waypts_vecs,
            input_keys=waypts_names,
            output_keys=waypts_names,
            wta_output=True)
        nengo.Connection(model.prev_waypt.output, model.waypts_prev2vec.input)

        # Create state representing normalized vector to the next waypoint
        model.norm_waypt_vec = spa.State(2, vocab=v_norm_waypts_vecs)
        nengo.Connection(model.waypts_prev2vec.output,
                         model.norm_waypt_vec.input)

        # Create ensemble representing normalized vector to the next waypoint
        # relative to the robot
        bot_norm_waypt_loc_vec_interm = nengo.Ensemble(400, dimensions=4)
        nengo.Connection(model.norm_waypt_vec.output,
                         bot_norm_waypt_loc_vec_interm[0:2], synapse=None)
        nengo.Connection(
            bot_orient_inp, bot_norm_waypt_loc_vec_interm[2:4], synapse=None,
            function=lambda x: (np.cos(-x), np.sin(-x)))

        def calc_bot_norm_waypt_loc_vec(x):
            waypt_rot = np.array([[x[2], -x[3]], [x[3], x[2]]], dtype=float)
            return np.dot(waypt_rot, x[0:2])

        bot_norm_waypt_loc_vec = nengo.Ensemble(100, dimensions=2)
        nengo.Connection(bot_norm_waypt_loc_vec_interm, bot_norm_waypt_loc_vec,
                         function=calc_bot_norm_waypt_loc_vec)

        # Create associative memory between previous and next waypoints
        model.waypts_prev2next = spa.AssociativeMemory(
            input_vocab=v_waypts,
            output_vocab=v_waypts,
            input_keys=waypts_names,
            output_keys=waypts_names[1:]+waypts_names[-1:],
            wta_output=True)
        nengo.Connection(model.prev_waypt.output, model.waypts_prev2next.input)

        # Create state representing next waypoint
        model.next_waypt = spa.State(params.sp_dim, vocab=v_waypts)
        nengo.Connection(model.waypts_prev2next.output, model.next_waypt.input)

        # Create associative memory between waypoints and corresponding average
        # landmark vectors
        model.waypts2alvs = spa.AssociativeMemory(
            input_vocab=v_waypts,
            output_vocab=v_waypts_alvs,
            input_keys=waypts_names,
            output_keys=waypts_names,
            wta_output=True)
        nengo.Connection(model.next_waypt.output, model.waypts2alvs.input)

        # Create state representing average landmark vector at the next
        # waypoint
        model.next_waypt_alv = spa.State(2, neurons_per_dimension=500,
                                         vocab=v_waypts_alvs)
        nengo.Connection(model.waypts2alvs.output, model.next_waypt_alv.input)

        # Create node representing normalized distance from the previous
        # waypoint
        def calc_norm_prev_waypt_dist(t, x):
            bot_pos = np.array(x[0:2], dtype=float)
            w = np.argmax(nengo.spa.similarity(x[2:], v_waypts))
            if w < n_waypts - 1:
                dist = np.sqrt(((bot_pos - waypts_pos[w])**2).sum())
                return dist / waypts_dists[w]
            else:
                return 1.0

        norm_prev_waypt_dist_inp = nengo.Node(
            calc_norm_prev_waypt_dist, size_in=2+params.sp_dim, size_out=1)
        nengo.Connection(bot_pos_inp, norm_prev_waypt_dist_inp[0:2],
                         synapse=None)
        nengo.Connection(model.prev_waypt.output, norm_prev_waypt_dist_inp[2:])

        # Create ensemble representing gain of the difference between the
        # average landmark vector and the average landmark vector at the next
        # waypoint
        def calc_gain_diff_alv(x):
            if x[0] <= params.min_gain_diff_alv_dist:  # far from the next
                                                       # waypoint
                return 0.0
            elif x[0] >= params.max_gain_diff_alv_dist:  # close to next
                                                         # waypoint
                return 1.0
            else:  # not so far from the next waypoint
                return ((x[0] - params.min_gain_diff_alv_dist)
                        / diff_gain_diff_alv_dist)

        gain_diff_alv = nengo.Ensemble(100, dimensions=1)
        nengo.Connection(norm_prev_waypt_dist_inp, gain_diff_alv,
                         function=calc_gain_diff_alv)
        nengo.Connection(bot_alv_off, gain_diff_alv.neurons,
                         transform=[[-10.0]]*100)

        # Create ensemble representing difference between the average landmark
        # vector and the average landmark vector at the next waypoint
        diff_alv = nengo.Ensemble(2000, dimensions=2,
                                  radius=2*params.land_radius)
        nengo.Connection(bot_alv_inp[0:2], diff_alv, transform=1)
        nengo.Connection(model.next_waypt_alv.output, diff_alv, transform=-1)

        # Create ensemble representing thresholded magnitude of the difference
        # between the average landmark vector and the average landmark vector
        # at the next waypoint
        thres_diff_alv = nengo.Ensemble(
            1000, dimensions=1, encoders=nengo.dists.Choice([[1.0]]),
            intercepts=nengo.dists.Uniform(1.0 - 2 * params.diff_alv_thres,
                                           1.0))
        nengo.Connection(
            diff_alv, thres_diff_alv, synapse=0.05,
            function=lambda x: 1.0 - np.sqrt((x**2).sum()),
            eval_points=nengo.dists.Gaussian(0.0, params.diff_alv_thres))
        nengo.Connection(gain_diff_alv, thres_diff_alv, synapse=0.1,
                         function=lambda x: x - 1.0)
        nengo.Connection(bot_alv_off, thres_diff_alv, transform=-10.0)
        nengo.Connection(thres_diff_alv, thres_diff_alv, synapse=0.1,
                         transform=0.6)

        # Create ensemble representing velocity vector
        vel_vec_interm_bias_inp = nengo.Node(1.0)
        vel_vec_interm = nengo.networks.Product(200, dimensions=4,
                                                input_magnitude=1.0)
        nengo.Connection(gain_diff_alv, vel_vec_interm.A[0:2],
                         transform=[[1.0], [1.0]])
        nengo.Connection(gain_diff_alv, vel_vec_interm.A[2:4],
                         transform=[[-1.0], [-1.0]])
        nengo.Connection(vel_vec_interm_bias_inp, vel_vec_interm.A[2:4],
                         transform=[[1.0], [1.0]])
        nengo.Connection(diff_alv, vel_vec_interm.B[0:2])
        nengo.Connection(bot_norm_waypt_loc_vec, vel_vec_interm.B[2:4])

        vel_vec = nengo.Ensemble(200, dimensions=2)
        nengo.Connection(vel_vec_interm.output[[0,2]], vel_vec[0],
                         transform=[[1.0, 1.0]])
        nengo.Connection(vel_vec_interm.output[[1,3]], vel_vec[1],
                         transform=[[1.0, 1.0]])

        # Create ensemble representing wheel speeds
        def calc_wheel_speeds(x):
            if np.sqrt((x**2).sum()) < 0.05:
                return [0.0, 0.0]
            theta = np.arctan2(x[1], x[0])
            if -0.3 < theta < 0.3:
                return [5.0, 5.0]
            else:
                return [-5.0*theta, 5.0*theta]

        wheel_speeds = nengo.Ensemble(100, dimensions=2, radius=1.0)
        nengo.Connection(vel_vec, wheel_speeds, function=calc_wheel_speeds)
        vrep_comm.add_input(bot.wheels.set_velocities, 2)
        nengo.Connection(wheel_speeds, vrep_proxy[0:2])

        # Create action selection circuit
        model.thres_diff_alv_sta = spa.State(1, neurons_per_dimension=500,
                                             vocab=v_thres_diff_alv)
        nengo.Connection(thres_diff_alv, model.thres_diff_alv_sta.input,
                         synapse=0.1)
        actions = spa.Actions(
            'dot(thres_diff_alv_sta, REACHED) --> prev_waypt=0.9*next_waypt',
            '0.5 -->')
        model.bg = spa.BasalGanglia(actions)
        model.thal = spa.Thalamus(model.bg)

        # If necessary, create probes for recording dynamic data
        if save_data:
            bot_pos_prb = nengo.Probe(bot_pos_inp,
                                      sample_every=params.vrep_sim_dt)
            bot_orient_prb = nengo.Probe(bot_orient_inp,
                                         sample_every=params.vrep_sim_dt)
            bot_alv_prb = nengo.Probe(bot_alv_inp,
                                      sample_every=params.vrep_sim_dt)
            prev_waypt_prb = nengo.Probe(model.prev_waypt.output,
                                         synapse=params.prb_syn)
            prev_waypt_neurons_prb = nengo.Probe(
                model.prev_waypt.state_ensembles.add_neuron_output())
            next_waypt_prb = nengo.Probe(model.next_waypt.output,
                                         synapse=params.prb_syn)
            next_waypt_neurons_prb = nengo.Probe(
                model.next_waypt.state_ensembles.add_neuron_output())
            diff_alv_prb = nengo.Probe(diff_alv, synapse=params.prb_syn)
            gain_diff_alv_prb = nengo.Probe(gain_diff_alv,
                                            synapse=params.prb_syn)
            bg_prb = nengo.Probe(model.bg.input, synapse=params.prb_syn)
            thal_prb = nengo.Probe(model.thal.actions.output,
                                   synapse=params.prb_syn)
            thal_neurons_prb = nengo.Probe(
                model.thal.actions.add_neuron_output())

    # Start V-REP simulation in synchronous operation mode
    vrep_sim.start_sim(params.verbose)

    try:
        # Run simulation for the specified time or until the final waypoint is
        # reached
        nengo_sim = nengo.Simulator(model, dt=params.nengo_sim_dt)
        sim_time = 0.0
        with nengo_sim:
            while sim_time < params.sim_duration:
                # Retrieve robot position
                bot_pos = bot.get_position()[:2]

                # If the final waypoint has been reached, stop the wheels and
                # stop moving
                bot_final_waypt_dist = np.sqrt(
                    ((waypts_pos[-1] - bot_pos)**2).sum())
                if bot_final_waypt_dist <= params.final_waypt_dist_margin:
                    bot.wheels.set_velocities((0.0, 0.0))
                    if params.verbose:
                        print("Final waypoint {} reached."
                              "".format(waypts_names[-1]))
                        print("Simulation time: {:.3f} s.".format(sim_time))
                        print("Distance to waypoint: {:.2f} m."
                              "".format(bot_final_waypt_dist))
                    break

                # Run a single cycle of simulation
                nengo_sim.run(params.sim_cycle_duration)
                sim_time += params.sim_cycle_duration
            else:
                # If the final waypoint has not been reached until the time is
                # up, stop the wheels and stop moving
                bot.wheels.set_velocities((0.0, 0.0))
                if params.verbose:
                    print("Time up, final waypoint {} not reached."
                          "".format(waypts_names[-1]))
                    print("Distance to waypoint: {:.2f} m."
                          "".format(bot_final_waypt_dist))
    finally:
        # Stop V-REP simulation
        vrep_sim.stop_sim(params.verbose)

    # If necessary, save simulation data
    if save_data:
        # Save static data
        np.savetxt(os.path.join(data_dirname, waypts_pos_filename), waypts_pos,
                   "%.6f")
        np.savetxt(os.path.join(data_dirname, waypts_alvs_filename),
                   waypts_alvs, "%.6f")
        np.savetxt(os.path.join(data_dirname, waypts_names_filename),
                   waypts_names, "%s")
        np.savetxt(os.path.join(data_dirname, lands_pos_filename), lands_pos,
                   "%.6f")

        # Save dynamic data
        bot_pos_data = nengo_sim.data[bot_pos_prb]
        bot_orient_data = nengo_sim.data[bot_orient_prb] * (180 / np.pi)
        bot_alv_data = nengo_sim.data[bot_alv_prb]
        prev_waypt_data = model.similarity(nengo_sim.data, prev_waypt_prb)
        prev_waypt_neurons_data = nengo_sim.data[prev_waypt_neurons_prb]
        next_waypt_data = model.similarity(nengo_sim.data, next_waypt_prb)
        next_waypt_neurons_data = nengo_sim.data[next_waypt_neurons_prb]
        diff_alv_data = nengo_sim.data[diff_alv_prb]
        gain_diff_alv_data = nengo_sim.data[gain_diff_alv_prb]
        bg_data = nengo_sim.data[bg_prb]
        thal_data = nengo_sim.data[thal_prb]
        thal_neurons_data = nengo_sim.data[thal_neurons_prb]
        if not params.with_time:
            np.savetxt(os.path.join(data_dirname, bot_pos_filename),
                       bot_pos_data, "%.6f")
            np.savetxt(os.path.join(data_dirname, bot_orients_filename),
                       bot_orient_data, "%.4f")
            np.savetxt(os.path.join(data_dirname, bot_alvs_filename),
                       bot_alv_data, "%.6f")
            np.savetxt(os.path.join(data_dirname, prev_waypts_filename),
                       prev_waypt_data, "%.6f")
            np.savez_compressed(
                os.path.join(data_dirname, prev_waypt_neurons_filename),
                data=prev_waypt_neurons_data)
            np.savetxt(os.path.join(data_dirname, next_waypts_filename),
                       next_waypt_data, "%.6f")
            np.savez_compressed(
                os.path.join(data_dirname, next_waypt_neurons_filename),
                data=next_waypt_neurons_data)
            np.savetxt(os.path.join(data_dirname, diffs_alv_filename),
                       diff_alv_data, "%.6f")
            np.savetxt(os.path.join(data_dirname, gains_diff_alv_filename),
                       gain_diff_alv_data, "%.6f")
            np.savetxt(os.path.join(data_dirname, bg_filename), bg_data,
                       "%.6f")
            np.savetxt(os.path.join(data_dirname, thal_filename), thal_data,
                       "%.6f")
            np.savez_compressed(
                os.path.join(data_dirname, thal_neurons_filename),
                data=thal_neurons_data)
        else:
            t = nengo_sim.trange()[::sim_dt_ratio][:,np.newaxis]
            t_prec = "%.{}f".format(
                len(repr(params.nengo_sim_dt).split(".")[-1]))
            np.savetxt(os.path.join(data_dirname, bot_pos_filename),
                       np.concatenate((t, bot_pos_data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
            np.savetxt(os.path.join(data_dirname, bot_orients_filename),
                       np.concatenate((t, bot_orient_data), axis=1),
                       [t_prec, "%.4f"])
            np.savetxt(os.path.join(data_dirname, bot_alvs_filename),
                       np.concatenate((t, bot_alv_data), axis=1),
                       [t_prec, "%.6f", "%.6f", "%.6f"])
            t = nengo_sim.trange()[:,np.newaxis]
            np.savetxt(os.path.join(data_dirname, prev_waypts_filename),
                       np.concatenate((t, prev_waypt_data), axis=1),
                       [t_prec] + ["%.6f"] * prev_waypt_data.shape[1])
            np.savez_compressed(
                os.path.join(data_dirname, prev_waypt_neurons_filename),
                data=np.concatenate((t, prev_waypt_neurons_data), axis=1))
            np.savetxt(os.path.join(data_dirname, next_waypts_filename),
                       np.concatenate((t, next_waypt_data), axis=1),
                       [t_prec] + ["%.6f"] * next_waypt_data.shape[1])
            np.savez_compressed(
                os.path.join(data_dirname, next_waypt_neurons_filename),
                data=np.concatenate((t, next_waypt_neurons_data), axis=1))
            np.savetxt(os.path.join(data_dirname, diffs_alv_filename),
                       np.concatenate((t, diff_alv_data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
            np.savetxt(os.path.join(data_dirname, gains_diff_alv_filename),
                       np.concatenate((t, gain_diff_alv_data), axis=1),
                       [t_prec, "%.6f"])
            np.savetxt(os.path.join(data_dirname, bg_filename),
                       np.concatenate((t, bg_data), axis=1),
                       [t_prec] + ["%.6f"] * bg_data.shape[1])
            np.savetxt(os.path.join(data_dirname, thal_filename),
                       np.concatenate((t, thal_data), axis=1),
                       [t_prec] + ["%.6f"] * thal_data.shape[1])
            np.savez_compressed(
                os.path.join(data_dirname, thal_neurons_filename),
                data=np.concatenate((t, thal_neurons_data), axis=1))

        # Save parameters
        with open(os.path.join(data_dirname, params_filename), 'w') \
            as params_file:
            for paramname in saved_params:
                params_file.write("{0}: {1}\n".format(paramname,
                                                      params[paramname]))

        # Save software versions
        versions_info = [
            ('model', __version__),
            ('python', platform.python_version()),
            ('nengo', nengo.__version__),
            ('numpy', np.__version__),
            ('v-rep', vrep_sim.get_version()),
            ('vrepsim', vrs.__version__)
            ]
        with open(os.path.join(data_dirname, versions_filename), 'w') \
            as versions_file:
            for v in versions_info:
                versions_file.write("{0}: {1}\n".format(v[0], v[1]))

finally:
    # Disconnect from V-REP
    vrep_sim.disconnect(params.verbose)
