POMDP Beginner Tutorial {#tutorialpomdp}
========================================

This tutorial assumes you have read and understood the MDP tutorial.

This tutorial's code can be found in the `examples/POMDP/tiger_door.cpp` file,
including comments and additional nodes.

Another good tutorial on POMDPs available online can be found
[here](http://pomdp.org/tutorial/).

Partially Observable Markov Decision Process
--------------------------------------------

A Partially Observable Markov Decision Process, or POMDP in short, is a
generalization of the ideas of an MDP to environments where the agents does not
directly know the current state of the underlying environment. It simply has no
access to it.

Instead, after performing each action, the agent receives an **observation**
which helps it restrict the range of the possible current states. The
observation is obtained from the environment following an **observation
function**, which dictates the probability of obtaining a certain observation
given a certain state and action.

Indeed, a POMDP is defined as a tuple <*S*, *A*, *O*, *T*, *R*, *W*, *d*>, where:

- All previous MDP symbols maintain the same meaning.
- *O* is the set of possible observations the agent can receive.
- *W* is the observation function.

The agent thus, at any timestep, maintains a **belief** of which states the
environment is actually in: some will be more likely, and some less. The belief
is then a discrete probability distribution over all states, indicating what the
agent thinks likely.

In POMDP planning this belief can be maintained over time, as the agent knows
the POMDP dynamics. Thus, given his previous belief and knowing which action has
been performed and what observation was received, the agent can compute a new
belief, with probabilities for every possible state of the environment.

Given that this is a complicated process, POMDP policies are usually in the
shape of a tree. Depending on which action was performed, and which observation
was received, the agent descends the appropriate tree branch and sees what
action it should perform next. Thus, in POMDPs the horizon (the number of
timesteps that you plan to act in the environment) is very important, as it
significantly affects which actions you should take.

Keep in mind that POMDPs are much harder to solve than MDPs, so it's important
to keep the size of the state,action and observation spaces as small as
possible.

### POMDP Example ###

Let's try to create an POMDP to show how it would work out in practice. Our
problem is going to be the following:

> Suppose you are in front of two doors. Behind one is a treasure. Behind the
> other a man-eating tiger. You don't know in advance where each is, but you
> want to try to get to the treasure safely. Also, after picking, the doors are
> reset, so you can try again.
>
> You can try to listen for the breathing of the tiger. The world around you is
> somewhat noisy, so you may sometimes think you hear something coming from the
> wrong door.
>
> How much time should you spend listening for the tiger, before trying your
> luck and opening a door?

Let's think about how to encode this world into an POMDP. We will go through each
component of the POMDP and try to fill it out.

#### S ####

There are only two possibilities: either the door on the left holds the treasure
(and the tiger is behind the right), or vice-versa. This will be simple!

~~~{.cpp}
    size_t S = 2;
    enum {
        TIG_LEFT    = 0,
        TIG_RIGHT   = 1,
    };
~~~

#### A ####

Only three actions: open the door on the left, open the door on the right, or
wait and listen for the breathing tiger.

~~~{.cpp}
    size_t A = 3;
    enum {
        A_LISTEN = 0,
        A_LEFT   = 1,
        A_RIGHT  = 2,
    };
~~~

#### O ####

Finally, the observations: there are also only two. When we listen, we get an
observation on whether we heard the tiger behind the left door or the right one.

Keep in mind that such an observation does not mirror the truth! We may have
misheard, and it's this uncertainty that makes this problem a POMDP (rather than
an MDP).

~~~{.cpp}
    size_t O = 2;
    // Using same enum as states
~~~

#### T ####

The transition function in this problem is also quite simple. If we are
listening, nothing changes in the world. When we instead open a door, we
discover what's behind it, and the problem is reset.

We can directly write a 3-dimensional transition matrix, as this should be
relatively simple.

~~~{.cpp}
    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);

    // Transitions
    // If we listen, nothing changes.
    for ( size_t s = 0; s < S; ++s )
        transitions[s][A_LISTEN][s] = 1.0;

    // If we pick a door, tiger and treasure shuffle.
    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            transitions[s][A_LEFT ][s1] = 1.0 / S;
            transitions[s][A_RIGHT][s1] = 1.0 / S;
        }
    }
~~~

#### R ####

The reward function is similar. We want to give a small penalty for listening,
so that the agent won't try that forever. We'll give a decent reward for
obtaining a treasure, and a great penalty for opening the door to the tiger.

~~~cpp
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);

    // Rewards
    // Listening has a small penalty
    for ( size_t s = 0; s < S; ++s )
        for ( size_t s1 = 0; s1 < S; ++s1 )
            rewards[s][A_LISTEN][s1] = -1.0;

    // Treasure has a decent reward, and tiger a bad penalty.
    for ( size_t s1 = 0; s1 < S; ++s1 ) {
        rewards[TIG_RIGHT][A_LEFT][s1] = 10.0;
        rewards[TIG_LEFT ][A_LEFT][s1] = -100.0;

        rewards[TIG_LEFT ][A_RIGHT][s1] = 10.0;
        rewards[TIG_RIGHT][A_RIGHT][s1] = -100.0;
    }
~~~

#### W ####

Finally, the observation function. We want to give the agent an 85% chance of
hearing the tiger correctly. So most of the time the observation will mirror the
truth, but not always.

We also have to give the agent an observation when it opens a door; in that case
we return an observation randomly (with probability `1.0 / O`), so not to give
any information to the agent during a reset.

~~~cpp
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

    // Observations
    // If we listen, we guess right 85% of the time.
    observations[TIG_LEFT ][A_LISTEN][TIG_LEFT ] = 0.85;
    observations[TIG_LEFT ][A_LISTEN][TIG_RIGHT] = 0.15;

    observations[TIG_RIGHT][A_LISTEN][TIG_RIGHT] = 0.85;
    observations[TIG_RIGHT][A_LISTEN][TIG_LEFT ] = 0.15;

    // Otherwise we get no information on the environment.
    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t o = 0; o < O; ++o ) {
            observations[s][A_LEFT ][o] = 1.0 / O;
            observations[s][A_RIGHT][o] = 1.0 / O;
        }
    }
~~~

#### Discount ####

We finally use a discount of 0.95 as it's usually done with this problem. Its
meaning has not changed from the MDP case.

### The POMDP Model ###

Since, differently from the MDP tutorial, we have not written functions for T, R
and W, but instead we have directly created matrices, we can simply copy them
into a new `AIToolbox::POMDP::Model` to start solving.

The default POMDP model is an extension of a given MDP (since as we said, the
POMDP is a generalization of MDPs; the basic environment is still there, it's
just that the agent cannot see it directly). In our case, we inherit from
`AIToolbox::MDP::Model` to keep things simple.

~~~{.cpp}
    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);
    model.setDiscount(0.95);
~~~

#### The Actual Planning Code ####

In case of this POMDP, it's simple enough that we can solve it exactly. We use
the `AIToolbox::POMDP::IncrementalPruning` algorithm, which is one of the most
performant exact solvers available.

Here we show also some code that explains how to use the value function from the
solution to obtain a policy, and how to use it. As we mentioned, policies are
a bit more complex in POMDPs due to their tree-shape.

Please check out the documentation of `AIToolbox::POMDP::Policy` to better learn
how to use it.

~~~{.cpp}
    // Set the horizon. This will determine the optimality of the policy
    // dependent on how many steps of observation/action we plan to do. 1 means
    // we're just going to do one thing only, and we're done. 2 means we get to
    // do a single action, observe the result, and act again. And so on.
    unsigned horizon = 15;
    // The 0.0 is the tolerance factor, used with high horizons. It gives a way
    // to stop the computation if the policy has converged to something static.
    AIToolbox::POMDP::IncrementalPruning solver(horizon, 0.0);

    // Solve the model. After this line, the problem has been completely
    // solved. All that remains is setting up an experiment and see what
    // happens!
    auto solution = solver(model);

    // We create a policy from the solution, in order to obtain actual actions
    // depending on what happens in the environment.
    AIToolbox::POMDP::Policy policy(2, 3, 2, std::get<1>(solution));

    // We begin a simulation, we start from a uniform belief, which means that
    // we have no idea on which side the tiger is in. We sample from the belief
    // in order to get a "real" state for the world, since this code has to
    // both emulate the environment and control the agent. The agent won't know
    // the sampled state though, it will only have the belief to work with.
    AIToolbox::POMDP::Belief b(2); b << 0.5, 0.5;
    auto s = AIToolbox::sampleProbability(2, b, rand);

    // The first thing that happens is that we take an action, so we sample it now.
    auto [a, ID] = policy.sampleAction(b, horizon);

    // We loop for each step we have yet to do.
    double totalReward = 0.0;
    for (int t = horizon - 1; t >= 0; --t) {
        // We advance the world one step (the agent only sees the observation
        // and reward).
        auto [s1, o, r] = model.sampleSOR(s, a);
        // We and update our total reward.
        totalReward += r;

        // We explicitly update the agent belief. This is necessary in some
        // cases depending on convergence of the solution, see below.
        // It is a pretty expensive operation so if performance is required it
        // should be avoided.
        b = AIToolbox::POMDP::updateBelief(model, b, a, o);

        // Now we can use the observation to find out what action we should
        // do next.
        //
        // Depending on whether the solution converged or not, we have to use
        // the policy differently. Suppose that we planned for an horizon of 5,
        // but the solution converged after 3. Then the policy will only be
        // usable with horizons of 3 or less. For higher horizons, the highest
        // step of the policy suffices (since it converged), but it will need a
        // manual belief update to know what to do.
        //
        // Otherwise, the policy implicitly tracks the belief via the id it
        // returned from the last sampling, without the need for a belief
        // update. This is a consequence of the fact that POMDP policies are
        // computed from a piecewise linear and convex value function, so
        // ranges of similar beliefs actually result in needing to do the same
        // thing (since they are similar enough for the timesteps considered).
        if (t > (int)policy.getH())
            std::tie(a, ID) = policy.sampleAction(b, policy.getH());
        else
            std::tie(a, ID) = policy.sampleAction(ID, o, t);

        // Then we update the world
        s = s1;
    }
~~~

The full code of this example can be found in the `examples/POMDP/tiger_door.cpp`
file, and is built automatically by adding `-DMAKE_EXAMPLES=1` when running
CMake.

### Conclusions ###

This tutorial has given you a very brief introduction in the world of POMDPs.
Given that they are so hard to solve, a lot of research has been done in
approximate solvers: point-based solvers in particular, where the POMDP is
solved only for a certain number of possible beliefs, have seen great success
both in theory and in practical applications. AIToolbox implements some of them,
for example `AIToolbox::POMDP::PBVI`, `AIToolbox::POMDP::PERSEUS`, or even the
bound-based `AIToolbox::POMDP::GapMin`.

Remember to read each class' documentation, as they explain the ideas behind
each algorithm, and how to use them, and feel free to check out external
references to learn more about POMDPs.
