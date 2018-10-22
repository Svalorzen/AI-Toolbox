#ifndef AI_TOOLBOX_MDP_Q_GREEDY_POLICY_HEADER_FILE
#define AI_TOOLBOX_MDP_Q_GREEDY_POLICY_HEADER_FILE

#include <AIToolbox/MDP/Policies/QPolicyInterface.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class models a greedy policy through a QFunction.
     *
     * This class allows you to select effortlessly the best greedy actions
     * from a given QFunction.
     */
    class QGreedyPolicy : public QPolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param q The QFunction this policy is linked with.
             */
            QGreedyPolicy(const QFunction & q);

            /**
             * @brief This function chooses the greediest action for state s.
             *
             * If multiple actions would be equally as greedy, a random one
             * is returned.
             *
             * @param s The sampled state of the policy.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction(const size_t & s) const override;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * If multiple greedy actions exist, this function returns the
             * correct probability of picking each one, since we return a
             * random one with sampleAction().
             *
             * @param s The selected state.
             * @param a The selected action.
             *
             * @return This function returns 0 if the action is not greedy, and 1/the number of greedy actions otherwise.
             */
            virtual double getActionProbability(const size_t & s, const size_t & a) const override;

            /**
             * @brief This function returns a matrix containing all probabilities of the policy.
             *
             * Computing this function is approximately a bit more
             * efficient than calling repeatedly the getActionProbability()
             * function over and over, since it does not need to find out
             * the maxima of the underlying QFunction over and over.
             *
             * Ideally this function can be called only when there is a
             * repeated need to access the same policy values in an
             * efficient manner.
             */
            virtual Matrix2D getPolicy() const override;

        private:
            // To avoid reallocating a vector every time for sampling.
            mutable std::vector<size_t> bestActions_;
    };
}

#endif
