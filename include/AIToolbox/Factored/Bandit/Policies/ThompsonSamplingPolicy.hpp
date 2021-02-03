#ifndef AI_TOOLBOX_FACTORED_BANDIT_THOMPSON_SAMPLING_POLICY_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_THOMPSON_SAMPLING_POLICY_HEADER_FILE

#include <random>

#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Bandit/Experience.hpp>
#include <AIToolbox/Factored/Bandit/Policies/PolicyInterface.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class models a Thompson sampling policy.
     *
     * This class uses the Normal distribution in order to estimate its
     * certainty about each arm average reward. Thus, each arm is estimated
     * through a Normal distribution centered on the average for the arm, with
     * decreasing variance as more experience is gathered.
     *
     * Note that this class assumes that the reward obtained is normalized into
     * a [0,1] range (which it does not check).
     *
     * The usage of the Normal distribution best matches a Normally distributed
     * reward. Another implementation (not provided here) uses Beta
     * distributions to handle Bernoulli distributed rewards.
     */
    class ThompsonSamplingPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param exp The Experience we learn from.
             */
            ThompsonSamplingPolicy(const Experience & exp);

            /**
             * @brief This function chooses an action using Thompson sampling.
             *
             * For each possible local joint action, we sample its possible
             * value from a normal distribution with mean equal to its reported
             * Q-value and standard deviation equal to 1.0/(counts+1).
             *
             * We then perform VariableElimination on the produced rules to
             * select the optimal action to take.
             *
             * @return The chosen action.
             */
            virtual Action sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * WARNING: In this class the only way to compute the true
             * probability of selecting the input action is via numerical
             * integration, since we're dealing with |A| Normal random
             * variables. To avoid having to do this, we simply sample a lot
             * and return an approximation of the times the input action was
             * actually selected. This makes this function very very SLOW. Do
             * not call at will!!
             *
             * To keep things short, we call "sampleAction" 1000 times and
             * count how many times the provided input was sampled. This
             * requires performing 1000 VariableElimination runs.
             *
             * @param a The selected action.
             *
             * @return This function returns an approximation of the probability of choosing the input action.
             */
            virtual double getActionProbability(const Action & a) const override;

            /**
             * @brief This function constructs a graph by sampling the provided experience.
             *
             * This function is the core of ThompsonSamplingPolicy, and is
             * provided so that other methods can leverage Thompson sampling in
             * a simpler way.
             *
             * Given a newly built, empty graph, we sample the experience using
             * Student t-distribution, so that the values sampled for each
             * local joint action have the correct likelihood of being the true
             * ones, following the Bayesian posteriors.
             *
             * @param exp The experience data we need to use.
             * @param graph The output, constructed graph.
             * @param rnd The random engine needed to sample.
             */
            static void setupGraph(const Experience & exp, VariableElimination::GVE::Graph & graph, RandomEngine & rnd);

        private:
            const Experience & exp_;
    };
}

#endif
