# This example shows how to perform Value Iteration for a simple single
# agent game, modeled as MDP. It computes the value and Q-value, and shows for
# the optimal policy for sample states.
#
# Suppose you have a grid shaped world, of N x N cells by. The world loops
# on itself like a torus, so that the top and bottom borders are connected
# with each other, and the same is true for the left and right borders.
# Two creatures walk this world: a tiger, and an antelope. Both creatures
# can move in the following way: up, down, left or right or stand still.
# When they decide to move, their movement is deterministic. The two
# creatures have different goals.
#
# The goal of the antelope is to not get eaten by the tiger. However it is
# pretty clueless in doing so; in fact, it always moves or stands randomly,
# aside from when the tiger is directly next to it. In that case, it will
# move randomly anywhere, but towards the tiger.
#
# The tiger has the goal of catching the antelope. Once it catches it,
# the game ends. What would be the best way for it to move?

import argparse
import itertools
import os
import sys

build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', 'build')
if build_dir not in sys.path:
    sys.path.append(build_dir)

try:
    import MDP
except ImportError:
    raise ImportError("cannot find MDP, has it been built?")


def wrapDiff(coord1, coord2):
    """
    Compute difference between two coordinates, consistent with the
    wraparound world with size SQUARE_SIZE x SQUARE_SIZE.

    Parameters
    ----------
    coord1: int
        First coordinate.
    coord2: int
        Second coordinate.
    """
    diff = coord2 - coord1
    distance1 = abs(diff)
    distance2 = SQUARE_SIZE - distance1

    if distance1 < distance2:
        return diff
    elif diff > 0:
        return -distance2
    return distance2


def getTransitionProbability(coord1, action, coord2):
    """
    Given two coordinates and an action, return the transition probability.

    Parameters
    ----------
    coord1: tuple of int
        Four element tuple containing the current coordinates of the tiger and
        antelope.
    action: str
        String representing the tiger action, options are i
        'stand', 'up', 'down', 'left', 'right'.
    coord1: tuple of int
        Four element tuple containing the next coordinates of the tiger and
        antelope.
    """
    # We compute the distances traveled by both the antelope and the tiger.
    tiger_x1, tiger_y1, antelope_x1, antelope_y1 = coord1
    tiger_x2, tiger_y2, antelope_x2, antelope_y2 = coord2

    # First check whether they were both in the same cell before.
    # In that case the game would have ended, and nothing would happen anymore.
    # We model this as a self-absorbing state, or a state that always
    # transitions to itself.
    if (tiger_x1, tiger_y1) == (antelope_x1, antelope_y1) == \
            (tiger_x2, tiger_y2) == (antelope_x2, antelope_y2):
        return 1.0

    tigerMovementX = wrapDiff(tiger_x1, tiger_x2)
    tigerMovementY = wrapDiff(tiger_y1, tiger_y2)
    antelMovementX = wrapDiff(antelope_x1, antelope_x2)
    antelMovementY = wrapDiff(antelope_y1, antelope_y2)
    diffX = wrapDiff(tiger_x1, antelope_x1)
    diffY = wrapDiff(tiger_y1, antelope_y1)

    # Both the tiger and the antelope can only move by 1 cell max at each
    # timestep. Thus, if this is not the case, the transition is
    # impossible.
    if abs(tigerMovementX) + abs(tigerMovementY) > 1:
        return 0.0
    if abs(antelMovementX) + abs(antelMovementY) > 1:
        return 0.0
    # The tiger can move only in the direction specified by its action. If
    # it is not the case, the transition is impossible.
    if action == 'stand' and (tigerMovementX or tigerMovementY):
        return 0.0
    if action == 'up' and tigerMovementY != 1:
        return 0.0
    if action == 'down' and tigerMovementY != -1:
        return 0.0
    if action == 'left' and tigerMovementX != -1:
        return 0.0
    if action == 'right' and tigerMovementX != 1:
        return 0.0

    # If tiger and antelope are not adjacent, then the probability for any
    # move of the antelope is simply 1/5: it behaves randomly.
    if abs(diffX) + abs(diffY) > 1:
        return 1.0 / 5.0

    # Otherwise, first we check that the move was allowed, as
    # the antelope cannot move where the tiger was before.
    if tiger_x1 == antelope_x2 and tiger_y1 == antelope_y2:
        return 0.0

    # Then check whether they were somehow in the same cell?
    if diffX + diffY == 0:
        if coord1 == coord2:
            return 1.0
        return 0.0

    # Else the probability of this transition is 1 / 4, still random but
    # without a possible antelope action.
    return 1.0 / 4.0


def getReward(coord):
    """
    Given a state, return the reward.

    Parameters
    ----------
    coord: tuple of int
        Four element tuple containing the position of the tiger and antelope.
    """
    tiger_x, tiger_y, antelope_x, antelope_y = coord
    if tiger_x == antelope_x and tiger_y == antelope_y:
        return 10.0
    return 0.0


def encodeState(coord):
    """
    Convert from coordinate to state_index.

    Parameters
    ----------
    coord: tuple of int
        Four element tuple containing the position of the tiger and antelope.

    Returns
    -------
    state: int
        Index of the state.
    """
    state = 0
    multiplier = 1
    for c in coord:
        state += multiplier * c
        multiplier *= SQUARE_SIZE

    return state


def decodeState(state):
    """
    Convert from state_index to coordinate.

    Parameters
    ----------
    state: int
        Index of the state.

    Returns
    -------
    coord: tuple of int
        Four element tuple containing the position of the tiger and antelope.
    """
    coord = []
    for _ in range(4):
        c = state % SQUARE_SIZE
        state /= SQUARE_SIZE
        coord.append(c)
    return tuple(coord)


def draw_coord(coord):
    """
    Draw the grid world.

    Parameters
    ----------
    coord: tuple of int
        Four element tuple containing the position of the tiger and antelope.
    """
    t_x, t_y, a_x, a_y = coord

    print "y \ x",
    for x in range(SQUARE_SIZE):
        print " {}  ".format(x),
    print ""
    for y in range(SQUARE_SIZE-1, -1, -1):
        print " {}  ".format(y),
        for x in range(SQUARE_SIZE):
            agent = ' '
            if (t_x, t_y) == (x, y):
                agent = 'T'
            elif (a_x, a_y) == (x, y):
                agent = 'A'
            print "| {} ".format(agent),
        print "|"


def solve_mdp(horizon, epsilon):
    """
    Construct the gridworld MDP, and solve it using value iteration. Print the
    best found policy for sample states.

    Returns
    -------
    solution: tuple
    """
    print "Constructing MDP"

    # Statespace contains the tiger (x, y) and antelope (x, y). Note that
    # this is a very naive state representation: many of these states can be
    # aggregated! We leave this as an exercise to the reader :)
    # S = [(t_x, t_y, a_x, a_y), .. ]
    S = list(itertools.product(range(SQUARE_SIZE), repeat=4))

    # A = tiger actions
    A = ['stand', 'up', 'down', 'left', 'right']

    # T gives the transition probability for every s, a, s' triple.
    T = []
    for state in range(len(S)):
        # note that R can actually be constructed more efficiently, as it only
        # depends on the NEXT state
        coord = decodeState(state)
        T.append([[getTransitionProbability(coord, action,
                                            decodeState(next_state))
                   for next_state in range(len(S))] for action in A])

    # R gives the reward associated with every s, a, s' triple. In the current
    # example, we only specify reward for s', but we still need to give the
    # entire array.
    reward_row = [getReward(decodeState(next_state))
                  for next_state in range(len(S))]
    R = [[reward_row for _ in A] for _ in S]

    # set up the model
    model = MDP.Model(len(S), len(A))
    model.setTransitionFunction(T)
    model.setRewardFunction(R)

    # Perform value iteration
    print "Solving MDP using ValueIteration(horizon={}, epsilon={})".format(
        horizon, epsilon)

    solver = MDP.ValueIterationModel(horizon, epsilon)
    solution = solver(model)

    print "Converged:", solution[0]
    _, value_function, q_function = solution

    policy = MDP.QGreedyPolicy(q_function)

    # print the game state and policy for some interesting coordinates
    coords_of_interest = [(0, 0, 1, 1), (0, 0, 2, 2), (0, 0, 3, 3),
                          (1, 1, 1, 2)]
    for coord in coords_of_interest:
        state = encodeState(coord)
        print "\nCoord {}, state {}".format(coord, state)
        print "===============================\n"
        draw_coord(coord)

        print "\nPolicy:"
        for a_idx, action in enumerate(A):
            p_a_given_s = policy.getActionProbability(state, a_idx)
            print "  p({:>6} | s)={}".format(action, p_a_given_s)
    return solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--square-size', default=6, type=int,
                        help="Size of the square gridworld.")
    parser.add_argument('-ho', '--horizon', default=100000, type=int,
                        help="Horizon parameter for value iteration")
    parser.add_argument('-e', '--epsilon', default=0.01, type=float,
                        help="Epsilon parameter for value iteration")

    args = parser.parse_args()
    SQUARE_SIZE = args.square_size
    solve_mdp(horizon=args.horizon, epsilon=args.epsilon)
