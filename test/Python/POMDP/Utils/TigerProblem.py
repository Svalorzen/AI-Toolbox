from AIToolbox import POMDP
from builtins import range

A_LISTEN = 0
A_LEFT   = 1
A_RIGHT  = 2

TIG_LEFT    = 0
TIG_RIGHT   = 1

def makeTigerProblem():
    # Actions are: 0-listen, 1-open-left, 2-open-right
    S = 2
    A = 3
    O = 2

    model = POMDP.Model(O, S, A)

    transitions = [[[0 for x in range(S)] for y in range(A)] for k in range(S)]
    rewards = [[[0 for x in range(S)] for y in range(A)] for k in range(S)]
    observations = [[[0 for x in range(O)] for y in range(A)] for k in range(S)]

    # Transitions
    # If we listen, nothing changes.
    for s in range(S):
        transitions[s][A_LISTEN][s] = 1.0

    # If we pick a door, tiger and treasure shuffle.
    for s in range(S):
        for s1 in range(S):
            transitions[s][A_LEFT ][s1] = 1.0 / S
            transitions[s][A_RIGHT][s1] = 1.0 / S

    # Observations
    # If we listen, we guess right 85% of the time.
    observations[TIG_LEFT ][A_LISTEN][TIG_LEFT ] = 0.85
    observations[TIG_LEFT ][A_LISTEN][TIG_RIGHT] = 0.15

    observations[TIG_RIGHT][A_LISTEN][TIG_RIGHT] = 0.85
    observations[TIG_RIGHT][A_LISTEN][TIG_LEFT ] = 0.15

    # Otherwise we get no information on the environment.
    for s in range(S):
        for o in range(O):
            observations[s][A_LEFT ][o] = 1.0 / O
            observations[s][A_RIGHT][o] = 1.0 / O

    # Rewards
    # Listening has a small penalty
    for s in range(S):
        for s1 in range(S):
            rewards[s][A_LISTEN][s1] = -1.0

    # Treasure has a decent reward, and tiger a bad penalty.
    for s1 in range(S):
        rewards[TIG_RIGHT][A_LEFT][s1] = 10.0
        rewards[TIG_LEFT ][A_LEFT][s1] = -100.0

        rewards[TIG_LEFT ][A_RIGHT][s1] = 10.0
        rewards[TIG_RIGHT][A_RIGHT][s1] = -100.0

    model.setTransitionFunction(transitions)
    model.setRewardFunction(rewards)
    model.setObservationFunction(observations)

    return model
