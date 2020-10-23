#ifndef AI_TOOLBOX_POMDP_TIGER_PROBLEM
#define AI_TOOLBOX_POMDP_TIGER_PROBLEM

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>

namespace AIToolbox::POMDP {
    namespace TigerProblemUtils {
        enum Action {
            A_LISTEN = 0,
            A_LEFT   = 1,
            A_RIGHT  = 2,
        };

        enum State {
            TIG_LEFT    = 0,
            TIG_RIGHT   = 1,
        };

        constexpr double listenError = 0.15;
    }

    /**
     * @brief This function sets up the tiger problem in a Model.
     *
     * This function builds the AAAI-94 Tiger problem, with
     * a 0.95 discount factor. The problem can be stated as follows:
     *
     * The agent stands in front of two doors. He can open either of
     * them. Behind one door, there is an agent-eater tiger, and
     * in the other a small treasure. The agent does not know to what
     * each door leads to, but instead of just opening the door, he
     * can listen. When he listens, it will hear the tiger from either
     * the left or right door. Its hearing is imperfect though, and
     * 15% of the time it will hear the tiger behind the wrong door.
     *
     * Once the agent opens a door, it will either get a great penalty
     * due to being eaten by the tiger, or get the reward. After that
     * the game will automatically reset to an unknown state: the agent
     * needs to start guessing again where the new tiger and treasure
     * are.
     *
     * The states here are the positions of the tiger and treasure:
     * since there are two doors, there are two states.
     *
     * There are three actions, corresponding to the listen action and
     * open door actions.
     *
     * There are two possible observations, which are always random but
     * for the listen action: in that case, we will obtain the correct
     * information about the true state 85% of the time.
     *
     * The solutions of this problem have been computed using Tony
     * Cassandra's pomdp-solve program (www.pomdp.org).
     *
     * @return The Model representing the problem.
     */
    inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> makeTigerProblem() {
        using namespace TigerProblemUtils;

        // Actions are: 0-listen, 1-open-left, 2-open-right
        constexpr size_t S = 2, A = 3, O = 2;

        AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

        AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
        AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
        AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

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

        // Observations
        // If we listen, we guess right 85% of the time.
        observations[TIG_LEFT ][A_LISTEN][TIG_LEFT ] = 1.0 - listenError;
        observations[TIG_LEFT ][A_LISTEN][TIG_RIGHT] = listenError;

        observations[TIG_RIGHT][A_LISTEN][TIG_RIGHT] = 1.0 - listenError;
        observations[TIG_RIGHT][A_LISTEN][TIG_LEFT ] = listenError;

        // Otherwise we get no information on the environment.
        for ( size_t s = 0; s < S; ++s ) {
            for ( size_t o = 0; o < O; ++o ) {
                observations[s][A_LEFT ][o] = 1.0 / O;
                observations[s][A_RIGHT][o] = 1.0 / O;
            }
        }

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

        model.setTransitionFunction(transitions);
        model.setRewardFunction(rewards);
        model.setObservationFunction(observations);

        return model;
    }
}

#endif
