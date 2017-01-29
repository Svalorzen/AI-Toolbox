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

We might as well add some code in order to determine the distance between two
coordinates, since we will need this later.

~~~{.cpp}
    constexpr int SQUARE_SIZE = 11;

    using CoordType = std::array<int, 4>;
    enum {
        TIGER_X = 0,
        TIGER_Y = 1,
        ANTEL_X = 2,
        ANTEL_Y = 3
    };

    // Returns distance between coordinates. It is consistent with
    // the wraparound world.
    int wrapDiff( int coord1, int coord2 ) {
        int diff = coord2 - coord1;

        int distance1 = std::abs( diff ), distance2 = SQUARE_SIZE - distance1;
        if ( distance1 < distance2 ) return diff;
        return diff > 0 ? -distance2 : distance2;
    }
~~~

#### A ####
The tiger can move, and possibly stand still. Thus, it has 5 actions.

~~~{.cpp}
    size_t A = 5;
    enum {
        UP    = 0,
        DOWN  = 1,
        LEFT  = 2,
        RIGHT = 3,
        STAND = 4
    };
~~~

#### T ####
Transition functions are generally the most time consuming part of defining an
MDP, and where most can go wrong. Most exact MDP solving methods rely on the
full transition function to find out the best policy for the problem. However,
you do not need to manually create a table containing all of them, as long as
you can compute them on the fly. This however can become *very* expensive
computationally; how much of the transition function you want to cache is your
decision to make.

In this function we specify the probability of ending up in a certain
tiger-antelope state, given that the tiger has taken a certain action from a
certain initial state.

~~~{.cpp}
    double getTransitionProbability( const CoordType & c1, size_t action, const CoordType & c2 ) {
        // We compute the distances traveled by both the antelope and the tiger.
        int tigerMovementX = wrapDiff( c1[TIGER_X], c2[TIGER_X] );
        int tigerMovementY = wrapDiff( c1[TIGER_Y], c2[TIGER_Y] );
        int antelMovementX = wrapDiff( c1[ANTEL_X], c2[ANTEL_X] );
        int antelMovementY = wrapDiff( c1[ANTEL_Y], c2[ANTEL_Y] );

        // Both the tiger and the antelope can only move by 1 cell max at each
        // timestep. Thus, if this is not the case, the transition is
        // impossible.
        if ( std::abs( tigerMovementX ) +
             std::abs( tigerMovementY ) > 1 ) return 0.0;

        if ( std::abs( antelMovementX ) +
             std::abs( antelMovementY ) > 1 ) return 0.0;

        // The tiger can move only in the direction specified by its action. If
        // it is not the case, the transition is impossible.
        if ( action == STAND && ( tigerMovementX || tigerMovementY ) ) return 0.0;
        if ( action == UP    && tigerMovementY != 1  ) return 0.0;
        if ( action == DOWN  && tigerMovementY != -1 ) return 0.0;
        if ( action == LEFT  && tigerMovementX != -1 ) return 0.0;
        if ( action == RIGHT && tigerMovementX != 1  ) return 0.0;

        // Now we check whether the tiger was next to the antelope or not
        int diffX = wrapDiff( c1[TIGER_X], c1[ANTEL_X] );
        int diffY = wrapDiff( c1[TIGER_Y], c1[ANTEL_Y] );

        // If thew were not adjacent, then the probability for any move of the
        // antelope is simply 1/5: it behaves randomly.
        if ( std::abs( diffX ) + std::abs( diffY ) > 1 ) return 1.0 / 5.0;

        // Otherwise, first we check that the move was allowed, as
        // the antelope cannot move where the tiger was before.
        if ( c1[TIGER_X] == c2[ANTEL_X] && c1[TIGER_Y] == c2[ANTEL_Y] ) return 0.0;

        // As a last check, we check whether they were both in the same cell before.
        // In that case the game would have ended, and nothing would happen anymore.
        // We model this as a self-absorbing state, or a state that always transitions
        // to itself.
        if ( diffX + diffY == 0 ) {
            if ( c1 == c2 ) return 1.0;
            else return 0.0;
        }

        // Else the probability of this transition is 1 / 4, still random but without
        // a possible antelope action.
        return 1.0 / 4.0;
    }
~~~

#### R ####

Here we define the reward function. Fortunately for us, this can be done easily:
we will reward the tiger when it catches the antelope with a reward of 10, and
otherwise it will get no reward. In this particular example the amount of reward
the tiger obtains does not really matter, as long as it is positive, but 10
looks nice. When multiple reward situation are present, relative rewards start
to get an important role in the final policy of the agent!

~~~cpp
    double getReward( const CoordType & c ) {
        if ( c[TIGER_X] == c[ANTEL_X] && c[TIGER_Y] == c[ANTEL_Y] ) return 10.0;
        return 0.0;
    }
~~~

#### Discount ####

We want the tiger to catch the antelope. We want it to catch it no matter how
far it is, and at the same time we do not want it to wait forever to catch it.
To model this, a high discount value makes sense. A low discount value tends to
rapidly devalue reward far in the future, and it can happen, if the tiger takes
too much, that floating point errors will eat the reward before the solving
method can compute the best policy. A high discount value avoids this problem;
however, the time that the method takes to converge to the best solution
increases as the number of timesteps it must look in the future increases.

~~~{.cpp}
    constexpr double discount = 0.9;
~~~

### The MDP Model ###

The code we have wrote up until now is nearly all you need to compute the optimal
policy. There's just one thing more. AIToolbox works on model classes; it
expects them to have certain methods, and to work in certain ways. What you thus
need to do is to wrap the functionality we just wrote into a single class.

At the same time, one more thing needs to be addressed. Every problem has in
general a different type of state. This is problematic, because there's no
simple way, for example, to iterate over custom states. Since states are unique,
this library abstracts over this problem, and requires that states be integers.
In order to allow for the conversions of our states into integers, we can write
some code which will convert them.

~~~{.cpp}
    size_t encodeState(const CoordType & coords) {
        size_t state = 0; unsigned multiplier = 1;
        for ( auto c : coords ) {
            state += multiplier * c;
            multiplier *= SQUARE_SIZE;
        }
        return state;
    }

    CoordType decodeState(size_t state) {
        CoordType coords;
        for ( auto & c : coords ) {
            c = state % SQUARE_SIZE;
            state /= SQUARE_SIZE;
        }
        return coords;
    }
~~~

And finally, our wrapper:

~~~{.cpp}
    class GridWorld {
        public:
            // The number of possible states of our model is equal to all the
            // possible coordinates that the antelope and the tiger could have.
            size_t getS() const { return SQUARE_SIZE * SQUARE_SIZE * SQUARE_SIZE * SQUARE_SIZE; }
            // This function returns the number of available actions to the agent
            size_t getA() const { return ::A; }
            // This one gets the discount of the model
            double getDiscount() const { return ::discount; }

            double getTransitionProbability( size_t s, size_t a, size_t s1 ) const {
                return ::getTransitionProbability( decodeState( s ), a, decodeState( s1 ) );
            }

            // This function needs to take parameters as the transition one,
            // but we are lucky and our model only needs one of them to know
            // the reward!
            double getExpectedReward( size_t, size_t, size_t s1 ) const {
                return ::getReward( decodeState( s1 ) );
            }

            // These two functions are needed to keep template code in the library
            // simple, but you don't need to implement them for the method we use
            // in this example. Look into AIToolbox::MDP::is_generative_model and
            // AIToolbox::MDP::is_model to know more about this.
            std::tuple<size_t, double> sampleSR(size_t,size_t) const;
            bool isTerminal(size_t) const;
    };
~~~

Voilà! All is needed now is simply some AIToolbox magic!

~~~{.cpp}
    int main() {
        GridWorld world;

        // This is a method that solves MDPs completely. It has a couple of
        // parameters available, but in our case the defaults are perfectly
        // fine.
        AIToolbox::MDP::ValueIteration solver;

        std::cout << "Starting solver!\n";
        // This is where the magic happens. This could take around 10-20 minutes,
        // depending on your machine (most of the time is spent on this tutorial's
        // code, however, since it is a pretty inefficient implementation).
        // But you can play with it and make it better!
        auto solution = solver(world);

        std::cout << "Problem solved? " << std::get<0>(solution) << "\n";

        AIToolbox::MDP::Policy policy(world.getS(), world.getA(), std::get<1>(solution));

        std::cout << "Printing best actions when prey is in (5,5):\n\n";
        for ( int y = 10; y >= 0; --y ) {
            for ( int x = 0; x < 11; ++x ) {
                std::cout << policy.sampleAction( encodeState(CoordType{{x, y, 5, 5}}) ) << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\nSaving policy to file for later usage...\n";
        {
            // You can load up this policy again using ifstreams.
            // You will not need to solve the model again ever, and you
            // can embed the policy into any application you want!
            std::ofstream output("policy.txt");
            output << policy;
        }

        return 0;
    }
~~~

The full code of this example can be found in the `examples/MDP/tiger_antelope.cpp`
file, and can be build from there using `make` (given that you have already
built the library in folder `build/`).

### Conclusions ###

The code we saw was a very inefficient implementation, for a number of reasons.
First, the particular method we used needs to look up repeatedly the transition
probabilities of the MDP model we use. In our implementation, this needs to be
recomputed almost constantly. A better way would be to save them up into a
single transition matrix once, and simply return the values of the table when
asked. AIToolbox offers a pretty standard implementation for an MDP structured
in this way: AIToolbox::MDP::Model.

In addition, our state space was way bigger than what was actually needed. This
is because the problem is question has a very high symmetry. For once, it does
not actually matter where the antelope is, since we could simply translate both
the antelope and the tiger until the antelope is at coordinates 5,5. This we can
do because the world is toroidal.

Another thing is that the world is symmetrical, both vertically, horizontally and
diagonally. Thus we could rewrite the transition function and the model so that
only an eighth of the states are needed. Combined with the translational
symmetry, this would reduce enormously the time needed to solve it.

However, I hope it gave you enough on an introduction on the concepts that you
can start to play around with the library by yourself.
