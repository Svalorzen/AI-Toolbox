#ifndef AI_TOOLBOX_FACTORED_BANDIT_MULTI_AGENT_RMAX_POLICY_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_MULTI_AGENT_RMAX_POLICY_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>
#include <AIToolbox/Factored/Bandit/Experience.hpp>
#include <AIToolbox/Factored/Bandit/Policies/PolicyInterface.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class implements the MARMax bandit algorithm.
     *
     * This algorithm is used for best arm identification; in other words it
     * will explore so that after a certain time it will return the arm that it
     * thinks is the best with high confidence. Thus, MARMax does not care
     * about rewards (or costs) that are incurred along the way; the only goal
     * is to recommend the best arm as fast as possible.
     *
     * MARMax achieves this by using the counts for each local joint action,
     * along with an upper bound for the values of these local actions. Value
     * estimate are always initialized with their respective upper-bounds. Once
     * an action has been tried enough, its estimate is revised from the upper
     * bound to its empirical estimate.
     *
     * MARMax always pulls the highest value joint action given the current
     * estimates (including those fixed to the upper bound). Once the full
     * joint action to pull does not contain any upper bounds (meaning all
     * local components have been tried a certain number of times), it will
     * recommend it as the likely optimal.
     *
     * The number of timesteps required by MARMax to recommend an action depend
     * on its inital parameters: the upper bounds, along with the desired
     * tolerance (specified with epsilon) and probability of the recommendation
     * being correct (specified with delta). The tighter the required bounds,
     * the longer it is going to take.
     *
     * This class also models the MAVMax variant of MARMax, which is more
     * optimistic in its value estimates. This means that it will start
     * updating them from their upper bounds sooner, thus significantly
     * reducing the number of timesteps required before a recommendation can be
     * suggested. This optimistic method is set as the default, but can be
     * disabled by manually specifying a flag.
     *
     * This algorithm assumes all rewards are positive, as that is what its
     * theoretical bound is based on.
     */
    class MARMaxPolicy : public PolicyInterface {
        public:
            using Graph = VariableElimination::GVE::Graph;

            /**
             * @brief Basic constructor.
             *
             * The epsilon and delta parameters heavily influence the overall behavior of the algorithm.
             *
             * Specifically, the epsilon parameter specifies our tolerance for
             * sub-optimal joint actions. We consider correct recommending a
             * joint action when its (true) expected reward is greater or equal
             * than (1-epsilon) of the optimal.
             *
             * Thus, epsilon needs to be a value in the interval [0,1], where
             * with 0 we consider only the optimal action as an acceptable
             * recommendation, and 1 any action can be recommended.
             *
             * The delta parameter specifies our acceptable probabilistic
             * guarantee that our recommendation will satisfy our epsilon
             * constraint. Given that a bandit's returns are stochastic, it is
             * generally impossible to truly guarantee that an action has a
             * certain expected reward; instead we only guarantee that in
             * expectation our recommendation will be correct with probability
             * (1-delta).
             *
             * Thus, delta needs to be a value in the interval (0,1], where
             * with values near 0 we request a high certainty that the
             * recommendation is correct, and 1 we don't give any guarantee at
             * all.
             *
             * The optimistic parameter, if set to true, enables the MAVMax
             * variant of MARMax. This variant is more optimistic in updating
             * the estimated values of local actions, and should thus be
             * faster, while still respecting the input constraints.
             *
             * @param experience The Experience to use to select actions.
             * @param ranges The upper-bounds of all local rewards.
             * @param epsilon The parameter controlling our tolerance for sub-optimal actions.
             * @param delta The parameter controlling the probability of our final recommendation satisfying the epsilon constraint.
             * @param optimistic Whether we want our updates to be optimistic, i.e. the MAVMax variant.
             */
            MARMaxPolicy(const Experience & experience, Vector ranges, double epsilon, double delta, bool optimistic = true);

            /**
             * @brief This function returns the current action to take.
             *
             * The return value is pre-computed by the stepUpdateQ() function;
             * it just returns an internal caching variable.
             *
             * @return The full joint action which should be taken now.
             */
            virtual Action sampleAction() const override;

            /**
             * @brief This function updates the policy given the latest Experience changes.
             *
             * @param indeces The output of the Experience's record() method after feeding it new experience data.
             */
            void stepUpdateQ(const Experience::Indeces & indeces);

            /**
             * @brief This function returns whether a full joint action is ready to be recommended.
             */
            bool canRecommendAction() const;

            /**
             * @brief If a full joint action can be recommended, this function returns it.
             */
            Action recommendAction() const;

            /**
             * @brief This function returns either 0 or 1 depending on whether the input corresponds to the current output of sampleAction().
             */
            virtual double getActionProbability(const Action & a) const override;

            /**
             * @brief This function returns the currently set epsilon parameter.
             *
             * @return The currently set epsilon parameter.
             */
            double getEpsilon() const;

            /**
             * @brief This function returns the currently set delta parameter.
             *
             * @return The currently set delta parameter.
             */
            double getDelta() const;

            /**
             * @brief This function returns the internal m parameter.
             *
             * This parameter is computed on construction based on the
             * specified epsilon and delta parameters. It determines the number
             * of pulls required for a local arm to be considered fully
             * explored, at least in the context of respecting the requested
             * bounds.
             *
             * @return The internal m parameter.
             */
            unsigned getM() const;

            /**
             * @brief This function returns a reference to the underlying Experience we use.
             *
             * @return The internal Experience reference.
             */
            const Experience & getExperience() const;

        private:
            const Experience & exp_;
            Vector ranges_;
            double epsilon_, delta_;
            bool optimistic_;

            unsigned m_;

            QFunction values_;
            Graph graph_;

            bool canRecommend_;
            Action currentAction_;
    };
}

#endif
