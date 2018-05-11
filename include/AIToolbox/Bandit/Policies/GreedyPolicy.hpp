#ifndef AI_TOOLBOX_BANDIT_GREEDY_POLICY_HEADER_FILE
#define AI_TOOLBOX_BANDIT_GREEDY_POLICY_HEADER_FILE

#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class models a simple greedy policy.
     *
     * This class always selects the greediest action with respect to the
     * already obtained experience.
     */
    class GreedyPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param A The size of the action space.
             */
            GreedyPolicy(size_t A);

            /**
             * @brief This function updates the greedy policy based on the result of the action.
             *
             * We simply keep a rolling average for each action, which we
             * update here. The ones with the best average are the ones which
             * will be selected when sampling.
             *
             * @param a The action taken.
             * @param r The reward obtained.
             */
            void stepUpdateP(size_t a, double r);

            /**
             * @brief This function chooses the greediest action.
             *
             * If multiple actions would be equally as greedy, a random one
             * is returned.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * If multiple greedy actions exist, this function returns the
             * correct probability of picking each one, since we return a
             * random one with sampleAction().
             *
             * @param a The selected action.
             *
             * @return This function returns 0 if the action is not greedy, and 1/the number of greedy actions otherwise.
             */
            virtual double getActionProbability(const size_t & a) const override;

        private:
            // Average reward/tries per action
            std::vector<std::pair<double, unsigned>> experience_;
            // To avoid reallocating a vector every time for sampling.
            mutable std::vector<unsigned> bestActions_;
    };
}

#endif

