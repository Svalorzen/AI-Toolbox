#ifndef AI_TOOLBOX_POMDP_RTBSS_HEADER_FILE
#define AI_TOOLBOX_POMDP_RTBSS_HEADER_FILE

#include <limits>

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::POMDP {
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
     * belief. Note that values computed in different methods may differ
     * due to floating point approximation errors.
     */
    template <typename M>
    class RTBSS {
        static_assert(is_model_v<M>, "This class only works for POMDP models!");

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
    };

    template <typename M>
    RTBSS<M>::RTBSS(const M& m, const double maxR) :
            model_(m), S(model_.getS()), A(model_.getA()),
            O(model_.getO()), maxR_(maxR) {}

    template <typename M>
    std::tuple<size_t, double> RTBSS<M>::sampleAction(const Belief& b, const unsigned horizon) {
        maxA_ = 0; maxDepth_ = horizon;

        const double value = simulate(b, horizon);

        return std::make_tuple(maxA_, value);
    }

    template <typename M>
    double RTBSS<M>::simulate(const Belief & b, const unsigned horizon) {
        if ( horizon == 0 ) return 0;

        std::vector<size_t> actionList(A);

        // Here we use no heuristic to sort the actions. If you want one
        // add it here!
        std::iota(std::begin(actionList), std::end(actionList), 0);

        double max = -std::numeric_limits<double>::infinity();

        for ( auto a : actionList ) {
            double rew = beliefExpectedReward(model_, b, a);

            const double uBound = rew + upperBound(b, a, horizon - 1);
            if ( uBound > max ) {
                for ( size_t o = 0; o < O; ++o ) {
                    const auto nextBelief = updateBeliefUnnormalized(model_, b, a, o);
                    const double sum = nextBelief.sum();
                    // Only work if it makes sense
                    if ( checkDifferentSmall(sum, 0.0) )
                        rew += model_.getDiscount() * sum * simulate(nextBelief / sum, horizon - 1);
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
    double RTBSS<M>::upperBound(const Belief &, const size_t, const unsigned horizon) const {
        return model_.getDiscount() * maxR_ * horizon;
    }

    template <typename M>
    const M& RTBSS<M>::getModel() const {
        return model_;
    }
}

#endif
