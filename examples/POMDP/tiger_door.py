# -*- coding: utf-8 -*-

import time
from AIToolbox import MDP, POMDP

# RENDERING

# Special character to go back up when drawing.
up = list("\033[XA")
# Special character to go back to the beginning of the line.
back = list("\33[2K\r")

def goup(x):
    while x > 8:
        up[2] = '9'
        print "".join(up)
        x -= 8

    up[2] = str(x + 1)
    print "".join(up)

def godown(x):
    while x:
        print ""
        x -= 1

prize = [
    r"  ________  ",
    r"  |       |" + "\\",
    r"  |_______|/",
    r" / $$$$  /| ",
    r"+-------+ | ",
    r"|       |/  ",
    r"+-------+   ",
]

tiger = [
    r"            ",
    r"   (`/' ` | ",
    r"  /'`\ \   |",
    r" /<7' ;  \ " + "\\",
    r"/  _､-, `,-" + "\\",
    r"`-`  ､/ ;   ",
    r"     `-'    ",
]

closedDoor = [
    r"   ______   ",
    r"  /  ||  \  ",
    r" |   ||   | ",
    r" |   ||   | ",
    r" |   ||   | ",
    r" +===++===+ ",
    r"            ",
]

openDoor = [
    r"   ______   ",
    r"|\/      \/|",
    r"||        ||",
    r"||        ||",
    r"||        ||",
    r"||________||",
    r"|/        \|",
]

sound = [
    r"    -..-    ",
    r"            ",
    r"  '-,__,-'  ",
    r"            ",
    r" `,_    _,` ",
    r"    `--`    ",
    r"            ",
]

nosound = [
    r"            ",
    r"            ",
    r"            ",
    r"            ",
    r"            ",
    r"            ",
    r"            ",
]

# Different format for him!
man = [
    r"   ___   ",
    r"  //|\\  ",
    r"  \___/  ",
    r" \__|__/ ",
    r"    |    ",
    r"    |    ",
    r"   / \   ",
    r"  /   \  ",
]

# Random spaces to make the rendering look nice. Yeah this is ugly, but it's
# just for the rendering.
hspacer = "     "
manhspacer = ' ' * (len(hspacer) / 2 + len(prize[0]) - len(man[0]) / 2)
numspacer  = ' ' * ((len(prize[0]) - 8) / 2)

clockSpacer = numspacer + ' ' * ((len(hspacer) - 1) / 2)
strclock = r"/|\-"

# MODEL

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

    transitions = [[[0 for x in xrange(S)] for y in xrange(A)] for k in xrange(S)]
    rewards = [[[0 for x in xrange(S)] for y in xrange(A)] for k in xrange(S)]
    observations = [[[0 for x in xrange(O)] for y in xrange(A)] for k in xrange(S)]

    # Transitions
    # If we listen, nothing changes.
    for s in xrange(S):
        transitions[s][A_LISTEN][s] = 1.0

    # If we pick a door, tiger and treasure shuffle.
    for s in xrange(S):
        for s1 in xrange(S):
            transitions[s][A_LEFT ][s1] = 1.0 / S
            transitions[s][A_RIGHT][s1] = 1.0 / S

    # Observations
    # If we listen, we guess right 85% of the time.
    observations[TIG_LEFT ][A_LISTEN][TIG_LEFT ] = 0.85
    observations[TIG_LEFT ][A_LISTEN][TIG_RIGHT] = 0.15

    observations[TIG_RIGHT][A_LISTEN][TIG_RIGHT] = 0.85
    observations[TIG_RIGHT][A_LISTEN][TIG_LEFT ] = 0.15

    # Otherwise we get no information on the environment.
    for s in xrange(S):
        for o in xrange(O):
            observations[s][A_LEFT ][o] = 1.0 / O
            observations[s][A_RIGHT][o] = 1.0 / O

    # Rewards
    # Listening has a small penalty
    for s in xrange(S):
        for s1 in xrange(S):
            rewards[s][A_LISTEN][s1] = -1.0

    # Treasure has a decent reward, and tiger a bad penalty.
    for s1 in xrange(S):
        rewards[TIG_RIGHT][A_LEFT][s1] = 10.0
        rewards[TIG_LEFT ][A_LEFT][s1] = -100.0

        rewards[TIG_LEFT ][A_RIGHT][s1] = 10.0
        rewards[TIG_RIGHT][A_RIGHT][s1] = -100.0

    model.setTransitionFunction(transitions)
    model.setRewardFunction(rewards)
    model.setObservationFunction(observations)

    return model

if __name__ == "__main__":
    # Create model of the problem.
    model = makeTigerProblem()
    model.setDiscount(0.95)

    # Set the horizon. This will determine the optimality of the policy
    # dependent on how many steps of observation/action we plan to do. 1 means
    # we're just going to do one thing only, and we're done. 2 means we get to
    # do a single action, observe the result, and act again. And so on.
    horizon = 15
    # The 0.0 is the tolerance factor, used with high horizons. It gives a way
    # to stop the computation if the policy has converged to something static.
    solver = POMDP.IncrementalPruning(horizon, 0.0)

    # Solve the model. After this line, the problem has been completely
    # solved. All that remains is setting up an experiment and see what
    # happens!
    solution = solver(model)

    # We create a policy from the solution, in order to obtain actual actions
    # depending on what happens in the environment.
    policy = POMDP.Policy(2, 3, 2, solution[1])

    # We begin a simulation, we start from a uniform belief, which means that
    # we have no idea on which side the tiger is in. We sample from the belief
    # in order to get a "real" state for the world, since this code has to
    # both emulate the environment and control the agent. The agent won't know
    # the sampled state though, it will only have the belief to work with.
    b = [0.5, 0.5]
    s = 0

    # The first thing that happens is that we take an action, so we sample it now.
    a, ID = policy.sampleAction(b, horizon)

    # We loop for each step we have yet to do.
    totalReward = 0.0
    for t in xrange(horizon - 1, -1, -1):
        # We advance the world one step (the agent only sees the observation
        # and reward).
        s1, o, r = model.sampleSOR(s, a)
        # We update our total reward.
        totalReward += r

        # Rendering of the environment, depends on state, action and observation.
        left  = prize if s else tiger
        right = tiger if s else prize
        for i in xrange(len(prize)):
            print("%s%s%s" % (left[i], hspacer, right[i]))

        dleft  = (openDoor if a == A_LEFT  else closedDoor)
        dright = (openDoor if a == A_RIGHT else closedDoor)
        for i in xrange(len(prize)):
            print("%s%s%s" % (dleft[i], hspacer, dright[i]))

        sleft  = (sound if a == A_LISTEN and o == TIG_LEFT  else nosound)
        sright = (sound if a == A_LISTEN and o == TIG_RIGHT else nosound)
        for i in xrange(len(prize)):
            print("%s%s%s" % (sleft[i], hspacer, sright[i]))

        print("%s%s%s%s%s%s" % (numspacer, ("%.6f" % b[0]), clockSpacer,
                  strclock[t % len(strclock)],
                  clockSpacer, ("%.6f" % b[1])))

        for m in man:
            print("%s%s" % (manhspacer, m))

        print("Timestep missing: " + str(t) + "  ") # Print space after to clear
        print("Total reward:     " + str(totalReward) + "  ")

        goup(3 * len(prize) + len(man) + 3)

        # We explicitly update the belief to show the user what the agent is
        # thinking. This is also necessary in some cases (depending on
        # convergence of the solution, see below), otherwise its only for
        # rendering purpouses. It is a pretty expensive operation so if
        # performance is required it should be avoided.
        b = POMDP.updateBelief(model, b, a, o)

        # Now that we have rendered, we can use the observation to find out
        # what action we should do next.
        #
        # Depending on whether the solution converged or not, we have to use
        # the policy differently. Suppose that we planned for an horizon of 5,
        # but the solution converged after 3. Then the policy will only be
        # usable with horizons of 3 or less. For higher horizons, the highest
        # step of the policy suffices (since it converged), but it will need a
        # manual belief update to know what to do.
        #
        # Otherwise, the policy implicitly tracks the belief via the id it
        # returned from the last sampling, without the need for a belief
        # update. This is a consequence of the fact that POMDP policies are
        # computed from a piecewise linear and convex value function, so
        # ranges of similar beliefs actually result in needing to do the same
        # thing (since they are similar enough for the timesteps considered).
        if t > policy.getH():
            a, ID = policy.sampleAction(b, policy.getH())
        else:
            a, ID = policy.sampleAction(ID, o, t)

        # Then we update the world
        s = s1

        # Sleep 1 second so the user can see what is happening.
        time.sleep(1)

    # Put the cursor back where it should be.
    godown(3 * len(prize) + len(man) + 3)
