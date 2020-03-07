MDP RL Beginner Tutorial {#tutorialmdprl}
=========================================

This tutorial is meant to teach you how to learn a policy in an MDP even when
the dynamics are not known a priory.

This tutorial's code can be found in the `examples/MDP/cliff.cpp` file,
including comments and additional nodes.

Reinforcement Learning
----------------------

While exact definitions vary depending on who is asked, here we consider
reinforcement learning to be the process of learning a policy by interaction
with an environment, without having previous information about its dynamics.

In particular, the policy we want to learn is the one that maximizes the reward
the agent obtains from the environment. An additional constraint that we want to
impose is that we learn in in a way to maximize the reward we obtain during the
learning process. In this way, the exploration done by the agent is indirectly
directed at figuring out promising actions to perform, rather than simply trying
to understand the problem fully, which would require much more time.

There are many possible approaches to RL, and in this tutorial we focus on two:
model-free learning, where the agent directly learns a value function to direct
its actions, and model-based learning, where the agent tries to learn a model of
the environment and use it to plan its next steps.

In the first case, we don't try to learn what the environment does, and we only
focus on what the agent is doing. Some methods use the data to directly modify
the policy; in this tutorial instead we use the data to update a value-function,
which is then used to inform the agent's policy.

Otherwise, in the model-based case, we will need some infrastructure to store
the perceived transitions and rewards, as we want to build a model of the
environment. From this model we could plan to obtain a policy, but as our model
is updated every timestep (with the new data we gain), planning every timestep
gets very expensive. There are methods that can take advantage of the fact that
changes are incremental, and update their policies quite quickly.

### MDP RL Example ###

In this tutorial we won't create a model from scratch, and instead we will use
one of the environments that are provided in the AIToolbox library. The
environment is this:

> The agent stands in a gridworld, at the edge of a long cliff. He needs to walk
> towards some unknown point nearby, and it needs to figure out where it needs
> to go. At the same time, walking off the cliff sends the agent to its death,
> while resetting the trial.
> Can the agent figure out where to go, without falling into the cliff?

Keep in mind that while this description is useful for us, since we are doing RL
the agent itself will have no idea (initially) that it is in a gridworld, nor
that the actions we are going to give it are to move around the environment.
Everything will be unknown at first, and the agent will need to figure out
things as it explores around.

~~~{.cpp}
    // We make a gridworld of 12 width and 3 height, using AIToolbox provided
    // class.
    GridWorld grid(12, 3);

    // We then build a cliff problem out of it. The agent will start at the
    // bottom left corner of the grid, and its target will be the bottom right
    // corner. However, aside from this two corners, all the cells at the bottom
    // of the grid will be marked as the cliff: going there will give a large
    // penalty to the agent, and will reset its position to the bottom left corner.
    auto problem = makeCliffProblem(grid);
~~~

#### Q-Learning ####

For our model-free method, we are going to use `AIToolbox::MDP::QLearning`,
which is a staple of modern RL, for its flexibility, simplicity and power.

QLearning uses the data we obtain from interacting with the environment in order
to update a QFunction: a mapping between a state-action pair and a numerical
value. These values represent how good we think taking a given action in a given
state is: higher values are better.

From this QFunction we can then create a policy: in particular, we are going to
use a `AIToolbox::MDP::QGreedyPolicy`. This policy selects the action with the
highest value in a certain state: it acts *greedily* with respect to the
QFunction.

While always selecting the best action seems like a good idea, we need to
remember that here we are *learning*: at first, the agent has no idea of what
actions actually do!

So we combine this greedy policy with an `AIToolbox::MDP::EpsilonPolicy`, which
sometimes selects random actions to help the agent try out new things to see if
they work.

~~~{.cpp}
    // We create the QLearning method. It only needs to know the size of the
    // state and action space, and the discount of the problem (to correctly update
    // values).
    QLearning qlLearner(problem.getS(), problem.getA(), problem.getDiscount());

    // We get a reference to the QFunction that QLearning is updating, and use
    // it to construct a greedy policy.
    QGreedyPolicy gPolicy(qlLearner.getQFunction());

    // The greedy policy is then augmented with some randomness, to help the
    // agent explore. In this case, we are going to take random actions with
    // probability 0.1 (10%). In the other cases, we will ask the greedy policy
    // what to do, and return that.
    EpsilonPolicy ePolicy(gPolicy, 0.1);
~~~

Now, we need to write some code which will actually make the agent go around the
world and try out things. Differently from the planning tutorial, where the
problem would be solved in a single line of code, here we actually have to write
a loop to simulate the agents going around and observing the results of its
actions.

These observations will be passed to QLearning, which will automatically update
its QFunction, and by extension the policies (since the policies keep a
reference to the original QFunction, they don't have separate copies).

~~~{.cpp}
    // Initial starting point, the bottom left corner.
    size_t start = problem.getS() - 2;

    size_t s, a;
    // We perform 10000 episodes, which should be enough to learn this problem.
    // At the start of each episode, we reset the position of the agent. Note
    // that this reset is for the episode; if during the episode the agent falls
    // into the cliff it will also be reset.
    for ( int episode = 0; episode < 10000; ++episode ) {
        s = start;
        // We limit the length of the episode to 10000 timesteps, to prevent the
        // agent roaming around indefinitely.
        for ( int i = 0; i < 10000; ++i ) {
            // Obtain an action for this state (10% random, 90% what we think is
            // best to do given the current QFunction).
            a = ePolicy.sampleAction( s );

            // Sample a new state and reward from the problem
            const auto [s1, rew] = problem.sampleSR( s, a );

            // Pass the newly collected data to QLearning, to update the
            // QFunction and improve the agent's policies.
            qlLearner.stepUpdateQ( s, a, s1, rew );

            // If we reach the goal, the episode ends
            if ( s1 == problem.getS() - 1 ) break;

            s = s1;
        }
    }
~~~

Once we are done, the agent is ready to act optimally. The only difference is
that now we would draw the actions directly from the greedy policy, rather than
from the epsilon policy, to avoid taking random actions:

~~~{.cpp}
    // Take greedy actions directly, skipping ePolicy
    a = gPolicy.sampleAction( s );
~~~

#### Prioritized Sweeping ####

We're now going to try a different approach, where we try to learn a model of
the environment, and use that to obtain a policy.

First, we're going to need `AIToolbox::MDP::Experience`, which is a class that
record the data we observe from the environment. It records the number of
state-action to new state transitions, and for each state-action pair, the
average reward obtained, its standard deviation and M2 value (an aggregate of
the squared distance from the mean).

~~~{.cpp}
    Experience exp(problem.getS(), problem.getA());
~~~

Now, from this data, there are different ways to get a model. We're going to go
with the simplest, that is to use a maximum likelihood estimator to estimate the
transition and reward functions of the problem. Doing this is very simple:

~~~{.cpp}
    MaximumLikelihoodModel<Experience> learnedModel(exp, problem.getDiscount(), false);
~~~

Every time we update our Experience with new data, we'll have to manually sync
the learned model to update its transition and reward functions. This is not
done automatically since syncing can be a somewhat expensive operation, so
sometimes you may want to do it sporadically. Here this is not a concern though.

Now, this learnedModel is itself a model, so if we wanted to we could use
`AIToolbox::MDP::ValueIteration` to solve it and obtain a policy. However, since
we have not interacted with the environment yet, there is really nothing useful
to solve.

Instead, we create our learning method, `AIToolbox::MDP::PrioritizedSweeping`,
and policies:

~~~{.cpp}
    PrioritizedSweeping psLearner(learnedModel);

    QGreedyPolicy gPolicy(psLearner.getQFunction());
    EpsilonPolicy ePolicy(gPolicy, 0.1);
~~~

PrioritizedSweeping, similarly to QLearning, is an algorithm that keeps a
QFunction from the input model, and updates it as new data becomes available.
The difference with QLearning is that it is much more rapid with its updates, as
it is able to reason about the learned model to perform multiple updates for
each new data points (while QLearning only does one).

Additionally, note that we have created the QGreedyPolicy and EpsilonPolicy
again, so that we can make the agent act and explore in the world to gather
data.

Let's now setup the learning loop. Note that for PrioritizedSweeping we need
much fewer episodes, as we are extracting more information out of each sample,
and we are performing multiple updates to the QFunction per timestep, which
greatly speeds up learning.

~~~{.cpp}
    // Initial starting point, the bottom left corner.
    size_t start = problem.getS() - 2;

    size_t s, a;
    // We perform 100 episodes, which should be enough to learn this problem.
    // Note that PrioritizedSweeping needs much fewer episodes to learn
    // effectively, as it is using the learned model to extract as much
    // information as possible and doing many updates per timestep.
    // At the start of each episode, we reset the position of the agent. Note
    // that this reset is for the episode; if during the episode the agent falls
    // into the cliff it will also be reset.
    for ( int episode = 0; episode < 100; ++episode ) {
        s = start;
        // We limit the length of the episode to 10000 timesteps, to prevent the
        // agent roaming around indefinitely.
        for ( int i = 0; i < 10000; ++i ) {
            // Obtain an action for this state (10% random, 90% what we think is
            // best to do given the current QFunction).
            a = ePolicy.sampleAction( s );

            // Sample a new state and reward from the problem
            const auto [s1, rew] = problem.sampleSR( s, a );

            // Record the new data in the Experience, so we can track it
            exp.record(s, a, s1, rew);

            // Update the learned model with the data we have just got.
            // This updates both the transition and reward functions.
            learnedModel.sync(s, a, s1);

            // Update the QFunction using this data.
            psLearner.stepUpdateQ(s, a);
            // Finally, use PrioritizedSweeping reasoning capabilities in order
            // to perform additional updates, and learn much more rapidly that
            // QLearning.
            psLearner.batchUpdateQ();

            // If we reach the goal, the episode ends
            if ( s1 == problem.getS() - 1 ) break;

            s = s1;
        }
    }
~~~

As you can see, we didn't need many more lines of code in order to run this
model-based method.

The full code of this example can be found in the `examples/MDP/cliff.cpp`
file, and can be build from there using `make` (given that you have already
built the library in folder `build/`).

### Conclusions ###

This tutorial should have given you the basics to start using the RL tools in
AIToolbox. Most other algorithms and classes work in a similar manner to the
ones described, so it should not be too difficult to read the documentation and
understand how they work.

Remember that if the documentation is unclear or you need help you can always
open an issue on GitHub!
