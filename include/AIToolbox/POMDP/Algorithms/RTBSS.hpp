#ifndef AI_TOOLBOX_POMDP_POMCP_HEADER_FILE
#define AI_TOOLBOX_POMDP_POMCP_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>

#include <limits>

namespace AIToolbox {
    namespace POMDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
        class RTBSS;
#endif

        /**
         * @brief This class represents the RTBSS online planner.
         *
         * This algorithm is an online planner for POMDPs. It works by pretty
         * much solving the whole POMDP in a straightforward manner, but just
         * for the belief it is currently in, and the horizon specified.
         *
         * Additionally, it uses an heuristic function in order to prune
         * branches which cannot possibly help in determining which action is
         * the actual best. Currently this heuristic is very crude, as it
         * requires the user to manually input a maximum possible reward, and
         * using it as an upper bound.
         *
         * Additionally, in theory one would want to explore branches from the
         * most promising to the least promising, to maximize pruning. This is
         * currently not done here, since an heuristic is intrinsically
         * determined by a particular problem. At the same time, it is easy to
         * add one, as the code specifies where one should be inserted.
         *
         * This method is able to return not only the best available action,
         * but also the (in theory) true value of that action in the current
         * belief.  Note that values computed in different methods may differ
         * due to floating point approximation errors.
         */
        template <typename M>
        class RTBSS<M> {
            public:

                /**
                 * @brief Basic constructor.
                 *
                 * @param m The POMDP model that POMCP will operate upon.
                 * @param maxR The max reward obtainable in the model. This is used for the pruning heuristic.
                 */
                RTBSS(const M& m, double maxR);

                /**
                 * @brief This function computes the best value for a given belief and its value.
                 *
                 * @param b The initial belief for the environment.
                 * @param horizon The horizon to plan for.
                 *
                 * @return The best action and its value in the model.
                 */
                std::tuple<size_t, double> sampleAction(const Belief& b, unsigned horizon);

                /**
                 * @brief This function returns the POMDP model being used.
                 *
                 * @return The POMDP model.
                 */
                const M& getModel() const;

            private:
                const M& model_;
                size_t S, A, O;
                size_t maxA_, maxDepth_;
                double maxR_;

                /**
                 * @brief This function performs the actual work of computing the best action and its value.
                 *
                 * Note that the best action is saved through the class variable
                 * maxA_, and is not directly returned here. This is mostly because
                 * knowing the best action is only useful at the top level, and
                 * not in the lower levels.
                 *
                 * @param b The belief to plan for.
                 * @param horizon The horizon to plan for.
                 *
                 * @return The value of the best action.
                 */
                double simulate(const Belief & b, unsigned horizon);

                /**
                 * @brief This function represents an heuristic to prune branches.
                 *
                 * This function is currently very crude, and it needs to be
                 * improved for your particular problem. The idea is to return
                 * the *future* reward that can be gained from a particular belief
                 * after performing a specific action (so it needs to be discounted).
                 *
                 * This upper bound must always overestimate the true value, but the
                 * closer it is to the true value the more pruning will be possible
                 * and the faster the method will run.
                 *
                 * @param b The belief from where we want to guess the future reward.
                 * @param a The action performed from the belief.
                 * @param horizon The timesteps remaining till the end.
                 *
                 * @return An overestimate of the reward that is possible to gain.
                 */
                double upperBound(const Belief & b, size_t a, unsigned horizon) const;


                /**
                 * @brief This function computes an immediate reward based on a belief rather than a state.
                 *
                 * @param b The belief to use.
                 * @param a The action performed from the belief.
                 *
                 * @return The immediate reward.
                 */
                double beliefReward(const Belief & b, size_t a) const;

                /**
                 * @brief This function computes the probability of obtaining an observation from a belief and action.
                 *
                 * @param b The belief to start from.
                 * @param a The action performed.
                 * @param o The observation that should be received.
                 *
                 * @return The probability of getting the observation from that belief and action.
                 */
                double beliefObservationProbability(const Belief & b, size_t a, size_t o) const;
        };

        template <typename M>
        RTBSS<M>::RTBSS(const M& m, double maxR) : model_(m), S(model_.getS()), A(model_.getA()), O(model_.getO()), maxR_(maxR) {}

        template <typename M>
        std::tuple<size_t, double> RTBSS<M>::sampleAction(const Belief& b, unsigned horizon) {
            maxA_ = 0; maxDepth_ = horizon;

            double value = simulate(b, horizon);

            return std::make_tuple(maxA_, value);
        }

        template <typename M>
        double RTBSS<M>::simulate(const Belief & b, unsigned horizon) {
            if ( horizon == 0 ) return 0;

            std::vector<size_t> actionList(A);

            // Here we use no heuristic to sort the actions. If you want one
            // add it here!
            std::iota(std::begin(actionList), std::end(actionList), 0);

            double max = -std::numeric_limits<double>::infinity();

            for ( auto a : actionList ) {
                double rew = beliefReward(b, a);

                double uBound = rew + upperBound(b, a, horizon - 1);
                if ( uBound > max ) {
                    for ( size_t o = 0; o < O; ++o ) {
                        double p = beliefObservationProbability(b, a, o);
                        // Only work if it makes sense
                        if ( p ) rew += model_.getDiscount() * p * simulate(updateBelief(model_, b, a, o), horizon - 1);
                    }
                }
                if ( rew > max ) {
                    max = rew;
                    if ( horizon == maxDepth_ ) maxA_ = a;
                }
            }
            return max;
        }

        template <typename M>
        double RTBSS<M>::upperBound(const Belief &, size_t, unsigned horizon) const {
            return model_.getDiscount() * maxR_ * horizon;
        }

        template <typename M>
        double RTBSS<M>::beliefReward(const Belief & b, size_t a) const {
            double rew = 0.0;
            for ( size_t s = 0; s < S; ++s )
                for ( size_t s1 = 0; s1 < S; ++s1 )
                    rew += model_.getTransitionProbability(s, a, s1) * model_.getExpectedReward(s, a, s1) * b[s];

            return rew;
        }

        template <typename M>
        double RTBSS<M>::beliefObservationProbability(const Belief & b, size_t a, size_t o) const {
            double p = 0.0;
            // This is basically the same as a belief update, but unnormalized
            // and we sum all elements together..
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                double sum = 0.0;
                for ( size_t s = 0; s < S; ++s )
                    sum += model_.getTransitionProbability(s, a, s1) * b[s];

                p += model_.getObservationProbability(s1, a, o) * sum;
            }

            return p;
        }

        template <typename M>
        const M& RTBSS<M>::getModel() const {
            return model_;
        }
    }
}

#endif
