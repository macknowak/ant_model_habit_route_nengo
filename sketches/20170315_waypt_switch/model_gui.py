"""Switching between waypoints along a route.

This script provides a Nengo model that proceeds from one waypoint to the next
along a route. Waypoints are represented as semantic pointers, and each
waypoint is associated with its successor as well as with the corresponding
average landmark vector. When the current waypoint is reached, a switch to the
next waypoint occurs, mediated by the action selection circuit. Reaching the
current waypoint is indicated by the status, which is also used during
initialization of the current waypoint. In both cases it must be set manually
using Nengo GUI.
"""

__author__ = "Przemyslaw (Mack) Nowak, Terry Stewart"

import nengo
import nengo.spa as spa

D = 32

# Create vocabularies
v_waypoint = spa.Vocabulary(D)
v_alv = spa.Vocabulary(2)
v_alv.add('ALV1', [0.3, 0.4])
v_status = spa.Vocabulary(D)

# Create Nengo model
model = spa.SPA()
with model:
    # Create state representing current waypoint
    model.waypoint = spa.State(D, vocab=v_waypoint, feedback=1)

    # Create associative memory between waypoints and average landmark vectors
    model.w2a = spa.AssociativeMemory(
        input_vocab=v_waypoint,
        output_vocab=v_alv,
        input_keys=['W1', 'W2', 'W3'],
        output_keys=['ALV1', 'ALV2', 'ALV3'],
        wta_output=True,
        )
    nengo.Connection(model.waypoint.output, model.w2a.input)

    # Create state representing current average landmark vector
    model.alv = spa.State(2, vocab=v_alv)
    nengo.Connection(model.w2a.output, model.alv.input)

    # Create associative memory between current and next waypoints
    model.w2next = spa.AssociativeMemory(
        input_vocab=v_waypoint,
        output_vocab=v_waypoint,
        input_keys=['W3', 'W2', 'W1'],
        output_keys=['W2', 'W1', 'W1'],
        wta_output=True)
    nengo.Connection(model.waypoint.output, model.w2next.input)

    # Create state representing next waypoint
    model.next_w = spa.State(D, vocab=v_waypoint)
    nengo.Connection(model.w2next.output, model.next_w.input)

    # Create state representing current status
    model.status = spa.State(D, vocab=v_status)

    # Create action selection circuit
    actions = spa.Actions(
        'dot(status, INIT) --> waypoint=W3',
        'dot(status, ARRIVED) --> waypoint=0.5*next_w',
        '0.5 -->'
        )
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)
