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
import time
from random import randint

build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'build')
if build_dir not in sys.path:
    sys.path.append(build_dir)

try:
    from AIToolbox import MDP
except ImportError:
    raise ImportError("cannot find MDP, has it been built?")

# Model

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
    coord2: tuple of int
        Four element tuple containing the next coordinates of the tiger and
        antelope.
    """
    # We compute the distances traveled by both the antelope and the tiger.
    tiger_x1, tiger_y1, antelope_x1, antelope_y1 = coord1
    tiger_x2, tiger_y2, antelope_x2, antelope_y2 = coord2

    # We compute the distances traveled by both the antelope and the tiger.
    tigerMovementX = wrapDiff(tiger_x1, tiger_x2)
    tigerMovementY = wrapDiff(tiger_y1, tiger_y2)
    antelMovementX = wrapDiff(antelope_x1, antelope_x2)
    antelMovementY = wrapDiff(antelope_y1, antelope_y2)

    # Both the tiger and the antelope can only move by 1 cell max at each
    # timestep. Thus, if this is not the case, the transition is
    # impossible.
    if abs(tigerMovementX) + abs(tigerMovementY) > 1:
        return 0.0
    if abs(antelMovementX) + abs(antelMovementY) > 1:
        return 0.0

    # Now we check whether the tiger was next to the antelope or not
    diffX = wrapDiff(tiger_x1, antelope_x1)
    diffY = wrapDiff(tiger_y1, antelope_y1)

    # We check whether they were both in the same cell before.
    # In that case the game would have ended, and nothing would happen anymore.
    # We model this as a self-absorbing state, or a state that always
    # transitions to itself. This is valid no matter the action taken.
    if diffX == 0 and diffY == 0:
        if coord1 == coord2:
            return 1.0
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

    Returns
    -------
    reward: float
        If the tiger catches the antelope, reward is +10. If not, reward is 0.0
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


#RENDERING

# Special character to go back up when drawing.
up = list("\033[XA")
# Special character to go back to the beginning of the line.
back = list("\33[2K\r")

def goup(x):
    """ Moves the cursor up by x lines """
    while x > 8:
        up[2] = '9'
        print "".join(up)
        x -= 8

    up[2] = str(x + 1)
    print "".join(up)

def godown(x):
    """ Moves the cursor down by x lines """
    while x:
        print ""
        x -= 1

def printState(coord):
    """
    Draw the grid world.

    - @ represents the tiger.
    - A represents the antelope.

    Parameters
    ----------
    coord: tuple of int
        Four element tuple containing the position of the tiger and antelope.
    """
    t_x, t_y, a_x, a_y = coord
    for y in range(SQUARE_SIZE-1, -1, -1):
        for x in range(SQUARE_SIZE):
            if (t_x, t_y) == (x, y):
                print "@",
            elif (a_x, a_y) == (x, y):
                print "A",
            else:
                print ".",
        print ""

def solve_mdp(horizon, tolerance, discount=0.9):
    """
    Construct the gridworld MDP, and solve it using value iteration. Print the
    best found policy for sample states.

    Returns
    -------
    solution: tuple
        First element is a boolean that indicates whether the method has
        converged. The second element is the value function. The third
        element is the Q-value function, from which a policy can be derived.
    """
    print time.strftime("%H:%M:%S"), "- Constructing MDP..."

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
        coord = decodeState(state)
        T.append([[getTransitionProbability(coord, action,
                                            decodeState(next_state))
                   for next_state in range(len(S))] for action in A])

    # R gives the reward associated with every s, a, s' triple. In the current
    # example, we only specify reward for s', but we still need to give the
    # entire |S|x|A|x|S| array.
    reward_row = [getReward(decodeState(next_state))
                  for next_state in range(len(S))]
    R = [[reward_row for _ in A] for _ in S]

    # set up the model
    model = MDP.SparseModel(len(S), len(A))
    model.setTransitionFunction(T)
    model.setRewardFunction(R)
    model.setDiscount(discount)

    # Perform value iteration
    print time.strftime("%H:%M:%S"), "- Solving MDP using ValueIteration(horizon={}, tolerance={})".format(
        horizon, tolerance)

    solver = MDP.ValueIteration(horizon, tolerance)
    solution = solver(model)

    print time.strftime("%H:%M:%S"), "- Converged:", solution[0] < solver.getTolerance()
    _, value_function, q_function = solution

    policy = MDP.Policy(len(S), len(A), value_function)

    s = randint(0, SQUARE_SIZE**4 - 1)
    while model.isTerminal(s):
        s = randint(0, SQUARE_SIZE**4 - 1)

    totalReward = 0
    for t in xrange(100):
        printState(decodeState(s))

        if model.isTerminal(s):
            break

        a = policy.sampleAction(s)
        s1, r = model.sampleSR(s, a)

        totalReward += r
        s = s1

        goup(SQUARE_SIZE)

        state = encodeState(coord)

        # Sleep 1 second so the user can see what is happening.
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--square-size', default=5, type=int,
                        help="Size of the square gridworld.")
    parser.add_argument('-ho', '--horizon', default=1000000, type=int,
                        help="Horizon parameter for value iteration")
    parser.add_argument('-t', '--tolerance', default=0.001, type=float,
                        help="Tolerance parameter for value iteration")
    parser.add_argument('-d', '--discount', default=0.9, type=float,
                        help="Discount parameter for value iteration")

    args = parser.parse_args()
    SQUARE_SIZE = args.square_size
    solve_mdp(horizon=args.horizon, tolerance=args.tolerance)
