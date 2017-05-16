# Ant navigation along a single habitual route, controlled by Nengo

This Nengo model controls a differential drive robot simulated using V-REP
simulator, which mimics a desert ant approaching the nest along a habitual
route. Communication with V-REP is handled using the V-REP remote API.

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

The robot model used in V-REP is *Pioneer_p3dx*. It has several sensors, 2
motor-driven wheels (1 on the left and 1 on the right), and 1 caster wheel (in
the back). None of the sensors are used because robot position as well as
positions of landmarks are obtained directly. Each motor-driven wheel receives
signals controlling its velocity from the Nengo model.

To successfully run this model, V-REP simulator must first be launched with a
continuous remote API server service started and then scene file `scene.ttt`
must be opened in the simulator. To allow the model script to remotely control
the *Pioneer_p3dx* robot that is part of this scene, the original child script
associated with this robot was removed from the scene file.

Moreover, to successfully run this model, the following V-REP files (or,
alternatively, links to them) have to exist in the current directory so that
the V-REP remote API could be used (here, `VREP_DIR` denotes the directory in
which V-REP is installed):

- `vrep.py` (original file in:
  `VREP_DIR/programming/remoteApiBindings/python/python/`);
- `vrepConst.py` (original file in:
  `VREP_DIR/programming/remoteApiBindings/python/python/`);
- `[remoteApi.dll | remoteApi.dylib | remoteApi.so]` (original file in:
  `VREP_DIR/programming/remoteApiBindings/lib/lib/[32Bit | 64Bit]/`).

This model was developed for Nengo 2.4.0 and V-REP 3.4.0.
