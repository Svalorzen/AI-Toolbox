#ifndef AI_TOOLBOX_MDP_BANDIT_POLICY_ADAPTOR_HEADER_FILE
#define AI_TOOLBOX_MDP_BANDIT_POLICY_ADAPTOR_HEADER_FILE

#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>
#include <AIToolbox/MDP/Policies/PolicyInterface.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class extends a Bandit policy so that it can be called from MDP code.
     *
     * This class simply ignores all states that are passed to it, and just
     * uses the actions in order to sample and call the underlying Bandit code.
     *
     * @tparam BanditPolicy The Bandit policy to wrap.
     */
    template <typename BanditPolicy>
    class BanditPolicyAdaptor : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param s The size of the state space.
             * @param params The arguments for the underlying BanditPolicy.
             */
            template <typename... Args>
            BanditPolicyAdaptor(size_t s, Args&&... params);

            /**
             * @brief This function chooses a random action using the underlying bandit policy.
             *
             * @param s The unused sampled state of the policy.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction(const size_t & s) const override;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * @param s The unused selected state.
             * @param a The selected action.
             *
             * @return The probability of taking the selected action in the specified state.
             */
            virtual double getActionProbability(const size_t & s, const size_t & a) const override;

            /**
             * @brief This function returns a matrix containing all probabilities of the policy.
             *
             * This function returns a matrix replicating the Bandit policy for
             * every row.
             */
            virtual Matrix2D getPolicy() const override;

            /**
             * @brief This function returns a reference to the underlying BanditPolicy.
             */
            BanditPolicy & getBanditPolicy();

            /**
             * @brief This function returns a reference to the underlying BanditPolicy.
             */
            const BanditPolicy & getBanditPolicy() const;

        private:
            BanditPolicy policy_;
    };

    template <typename BP>
    template <typename... Args>
    BanditPolicyAdaptor<BP>::BanditPolicyAdaptor(const size_t s, Args&&... args) :
            Base(s, 1), policy_(std::forward<Args>(args)...)
    {
        // We need to fix this later since we can't initialize the policy
        // before Base.
        A = policy_.getA();
    }

    template <typename BP>
    size_t BanditPolicyAdaptor<BP>::sampleAction(const size_t &) const {
        return policy_.sampleAction();
    }

    template <typename BP>
    double BanditPolicyAdaptor<BP>::getActionProbability(const size_t &, const size_t & a) const {
        return policy_.getActionProbability(a);
    }

    template <typename BP>
    Matrix2D BanditPolicyAdaptor<BP>::getPolicy() const {
        return policy_.getPolicy().transpose().replicate(getS(), 1);
    }

    template <typename BP>
    BP & BanditPolicyAdaptor<BP>::getBanditPolicy() { return policy_; }

    template <typename BP>
    const BP & BanditPolicyAdaptor<BP>::getBanditPolicy() const { return policy_; }
}

#endif
