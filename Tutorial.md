AIToolbox Tutorial
==================

This document is meant to give you information about this library and the
theoretical foundations behind it. It is not meant to be a full-on class about
it, but I hope it will be enough to get you started.

Markov Decision Process
-----------------------

A Markov Decision Process, or MDP in short, is a way to model a decision making
process. Here we will focus on single agent decision making processes, since
they are the most simple ones, and also the ones that this library is currently
focusing.

Let's start with some definitions:

- All starts from the **environment**. The environment is the setting of our
  decision making process. The environment changes over discrete timesteps; at
  each timestep the environment is in one, single, unique **state**. At each
  timestep it is said that the environment **transitions** from one state to
  another.
- In MDPs, it is said that states need to be **markovian**. What this means is
  that a state **must** encode all the information necessary for the decision making
  process, without leaving anything aside. More precisely, the decision process
  must not need to look at the past in order to be effective. All information
  contained in the past history of the environment must be readily available
  from the current state. For example, time-dependent contingencies are not
  allowed. If a problem contains them, then timer-like counters must be embedded
  in the state for the modeling to work, so that a state always contain all the
  information needed and the past is not relevant for decisions.
- The way the environment transitions from a state to another is not necessarily
  deterministic. From a particular state, the environment can transition to any
  other state following a certain probability distribution (which CAN be
  deterministic).
- The goal of the decision making process is influence the way the environment
  transitions between states. In some sense, we have preferences between states,
  and we would like the environment to be in some state rather than another. We
  will encode that preference using **rewards**.
- The **agent** is the entity that is taking the decisions. It is not necessarily
  corporeal, nor does it actually have to be inside the environment; most times
  it can be useful to visualize it in such terms though (a unit moving through an
  environment, for example).
- The agent interacts with the environment through **actions**. An action
  modifies the way in which the environment transitions from a state to another.
  Thus, at each timestep, the agent needs to select the action which will
  maximize its obtained reward.
- An agent interacts with the environment using a **policy**. The policy is what
  tells the agent what actions to take in each state. A policy can be
  deterministic or stochastic.
- The agent is able to interact with the environment during a certain amount of
  timesteps. This is called the **horizon**. The horizon can be either finite,
  meaning that the agent will stop receiving rewards after a certain number of
  timesteps, and thus should think only for those, or infinite. In this last
  case the agent will keep interacting with the environment forever, and should
  thus plan accordingly.
- The last thing we need to define is the **discount**. This is a number between
  0 and 1, which determines how much rewards obtained in future timesteps affect
  the agent's decisions. A discount of 0 will make the agent greedy, as it will
  take actions that maximize the reward obtained in the next timestep, and
  nothing else. A discount of 1 will make the agent lazy, as it will delay
  reward-obtaining actions possibly indefinitely.

Armed with our new vocabulary, we can now define an MDP more formally. An MDP is
a tuple <*S*, *A*, *T*, *R*, *d*>, where:

- *S* is a set of states. This is basically a list of all possible states the
  environment can ever be in.
- *A* is a set of actions. This is a list of all the actions that an agent can
  take. Normally, in an MDP setting, we assume that the agent can select any
  actions all the time; as in, there are no states where some action is blocked.
- *T* is a **transition function**. This describes the way that the environment is
  allowed to evolve, and is, in essence, the description of your problem. It
  specifies for any triple <*s*, *a*, *s'*> the probability that the environment will
  transition from *s* to *s'*, if the agent selects action *a*.
- R is a **reward function**. Similarly, it contains the rewards that the agent
  will obtain, depending on how the environment transitions. These, differently
  from the transition function, are deterministic rewards. The reward function
  specifies for any triple <*s*, *a*, *s'*> the reward that the agent will
  obtain.
- d is the discount factor, which we discussed above.

### MDP Example ###

Let's try to create an MDP to show how it would work out in practice. Our
problem is going to be the following:

> Suppose you have a grid shaped world, of 11 cells by 11 cells. The world loops
> on itself like a torus, so that the top and bottom borders are connected with
> each other, and the same is true for the left and right borders. Two creatures
> walk this world: a tiger, and an antelope. Both creatures can move in the
> following way: up, down, left or right or stand still. When they decide to
> move, their movement is deterministic. The two creatures have different goals.
>
> The goal of the antelope is to not get eaten by the tiger. However it is
> pretty clueless in doing so; in fact, it always moves or stands randomly,
> aside from when the tiger is directly next to it. In that case, it will move
> randomly anywhere, but towards the tiger.
>
> The tiger has the goal of catching the antelope. Once it catches it, the game
> ends. What would be the best way for it to move?

Let's think about how to encode this world into an MDP. We will go through each
component of the MDP and try to fill it out.

#### S ####
There seem to be no time dependent components, so that makes it easier for us to
create the states. In this case a naive approach would be to use the current
coordinates of both the tiger and the antelope as our state. Each pair of
coordinate is unique and encodes completely all the information the tiger needs to act.

In the AIToolbox library, the way states are represented is via integer numbers.
Since each state is unique, it can be converted to a number. However, a good
thing is that in your code you do not need to use states in their integer
representation, but only when calling AIToolbox functions. We can write some
code to make this usage more easy:

    size_t S = 11 * 11 * 11 * 11; // Total number of states

    using CoordType = std::array<size_t, 4>;
    enum {
        TIGER_X = 0,
        TIGER_Y = 1,
        ANTEL_X = 2,
        ANTEL_Y = 3
    };

    size_t encodeState(const CoordType & coords) {
        size_t state = 0; unsigned multiplier = 1;
        for ( auto c : coords ) {
            state += multiplier * c;
            multiplier *= 11;
        }
        return state;
    }

    CoordType decodeState(size_t state) {
        CoordType coords;
        for ( auto & c : coords ) {
           c = state % 11;
           state /= 11;
        }
        return coords;
    }

#### A ####
The tiger can move, and possibly stand still. Thus, it has 5 actions.

    size_t A = 5;

    enum {
        UP    = 0,
        DOWN  = 1,
        LEFT  = 2,
        RIGHT = 3,
        STAND = 4
    };

#### T ####
Transition functions are generally the most time consuming part of defining an
MDP. Most exact MDP solving methods rely on the full transition function to find
out the best policy for the problem. However, you do not need to manually create
a table containing all of them, as long as you can compute them on the fly.

    double getTransitionProbability( const CoordType & c1, size_t action, const CoordType & c2 ) {
        int tigerMovementX = c1[TIGER_X] - c2[TIGER_X];
        int tigerMovementY = c1[TIGER_Y] - c2[TIGER_Y];
        int antelMovementX = c1[ANTEL_X] - c2[ANTEL_X];
        int antelMovementY = c1[ANTEL_Y] - c2[ANTEL_Y];

        // Both the tiger and the antelope can only move by 1 cell max at each
        // timestep. Thus, if this is not the case, the transition is
        // impossible.
        if ( std::abs( tigerMovementX ) +
             std::abs( tigerMovementY ) > 1 ) return 0.0;

        if ( std::abs( antelMovementX ) +
             std::abs( antelMovementY ) > 1 ) return 0.0;

        // The tiger can move only in the direction specified by its action. If
        // it is not the case, the transition is impossible.
        if ( action == STAND &&
