#ifndef AI_TOOLBOX_MDP_POLICY_WRAPPER_HEADER_FILE
#define AI_TOOLBOX_MDP_POLICY_WRAPPER_HEADER_FILE

#include <vector>
#include <tuple>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/Policies/PolicyInterface.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class provides an MDP Policy interface around a Matrix2D.
     *
     * This class reads from the input reference to a Matrix2D in order to
     * provide a simple interface to use a policy.
     *
     * This class exists so that you can handle your own policy matrix
     * efficiently. This class will NEVER check the consistency of the matrix,
     * so that is up to you.
     *
     * This class expects that the input matrix represents a valid probability,
     * so each row should sum up to one, and no element should be negative or
     * over one.
     *
     * If you are looking for a self-contained version of this class that can
     * more easily interact with the other classes in the library, look for
     * Policy.
     */
    class PolicyWrapper : public PolicyInterface {
        public:
            using PolicyMatrix = Matrix2D;

            /**
             * @brief Basic constructor.
             *
             * The input is assumed to contain a valid policy matrix!
             *
             * @param p The policy matrix to wrap.
             */
            PolicyWrapper(const PolicyMatrix & p);

            /**
             * @brief This function chooses a random action for state s, following the policy distribution.
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
             * @brief This function enables inspection of the internal policy.
             *
             * @return A constant reference to the internal policy.
             */
            const PolicyMatrix & getPolicyMatrix() const;

            /**
             * @brief This function returns a matrix containing all probabilities of the policy.
             *
             * This is simply a copy of the internal policy.
             *
             * WARNING: If you just want a reference to the internal policy, use getPolicyMatrix().
             */
            virtual Matrix2D getPolicy() const override;

        private:
            const PolicyMatrix & policy_;
    };
}

#endif
