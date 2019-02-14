#ifndef AI_TOOLBOX_MDP_WOLF_POLICY_HEADER_FILE
#define AI_TOOLBOX_MDP_WOLF_POLICY_HEADER_FILE

#include <vector>

#include <AIToolbox/MDP/Policies/QPolicyInterface.hpp>
#include <AIToolbox/MDP/Policies/PolicyWrapper.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class models the WoLF learning algorithm.
     *
     * What this algorithm does is it progressively modifies the policy
     * given changes in the underlying QFunction. In particular, it
     * modifies it rapidly if the agent is "losing" (getting less reward
     * than expected), and more slowly when "winning", since there's little
     * reason to change behaviour when things go right.
     *
     * An advantage of this algorithm is that it can allow the policy to
     * converge to non-deterministic solutions: for example two players
     * trying to outmatch each other in rock-paper-scissor. At the same
     * time, this particular version of the algorithm can take quite some
     * time to converge to a good solution.
     */
    class WoLFPolicy : public QPolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * See the setter functions to see what the parameters do.
             *
             * @param q The QFunction from which to extract policy updates.
             * @param deltaw The learning rate if this policy is currently winning.
             * @param deltal The learning rate if this policy is currently losing.
             * @param scaling The initial scaling rate to progressively reduce the learning rates.
             */
            WoLFPolicy(const QFunction & q, double deltaw = 0.0125, double deltal = 0.05, double scaling = 5000.0);

            /**
             * @brief This function updates the WoLF policy based on changes in the QFunction.
             *
             * This function should be called between agent's actions,
             * using the agent's current state.
             *
             * @param s The state that needs to be updated.
             */
            void stepUpdateP(size_t s);

            /**
             * @brief This function chooses an action for state s, following the policy distribution.
             *
             * Note that to improve learning it may be useful to wrap this policy into an EpsilonPolicy
             * in order to provide some exploration.
             *
             * @param s The sampled state of the policy.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction(const size_t & s) const override;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * @param s The selected state.
             * @param a The selected action.
             *
             * @return The probability of taking the selected action in the specified state.
             */
            virtual double getActionProbability(const size_t & s, const size_t & a) const override;

            /**
             * @brief This function returns a matrix containing all probabilities of the policy.
             *
             * Ideally this function can be called only when there is a
             * repeated need to access the same policy values in an
             * efficient manner.
             */
            virtual Matrix2D getPolicy() const override;

            /**
             * @brief This function sets the new learning rate if winning.
             *
             * This is the amount that the policy is modified when the updatePolicy() function is called
             * when WoLFPolicy determines that it is currently winning based on the current QFunction.
             *
             * @param deltaW The new learning rate during wins.
             */
            void setDeltaW(double deltaW);

            /**
             * @brief This function returns the current learning rate during winning.
             *
             * @return The learning rate during winning.
             */
            double getDeltaW() const;

            /**
             * @brief This function sets the new learning rate if losing.
             *
             * This is the amount that the policy is modified when the updatePolicy() function is called
             * when WoLFPolicy determines that it is currently losing based on the current QFunction.
             *
             * @param deltaL The new learning rate during loss.
             */
            void setDeltaL(double deltaL);

            /**
             * @brief This function returns the current learning rate during loss.
             *
             * @return The learning rate during loss.
             */
            double getDeltaL() const;

            /**
             * @brief This function modifies the scaling parameter.
             *
             * In order to be able to converge WoLFPolicy needs to progressively reduce the learning rates
             * over time. It does so automatically to avoid needing to call both learning rate setters
             * constantly. This is also because in theory the learning rate should change per state, so
             * it would be even harder to do outside.
             *
             * Once determined if the policy is winning or losing, the selected learning rate is scaled
             * with the following formula:
             *
             *     newLearningRate = originalLearningRate / ( c_[s] / scaling + 1 );
             *
             * @param scaling The new scaling factor.
             */
            void setScaling(double scaling);

            /**
             * @brief This function returns the current scaling parameter.
             *
             * @return The current scaling parameter.
             */
            double getScaling() const;

        private:
            double deltaW_, deltaL_, scaling_;

            std::vector<unsigned> c_;
            PolicyWrapper::PolicyMatrix avgPolicyMatrix_, actualPolicyMatrix_;
            PolicyWrapper avgPolicy_, actualPolicy_;
    };
}

#endif
