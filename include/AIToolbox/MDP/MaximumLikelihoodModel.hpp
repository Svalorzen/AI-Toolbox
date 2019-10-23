#ifndef AI_TOOLBOX_MDP_MAXIMUM_LIKELIHOOD_MODEL_HEADER_FILE
#define AI_TOOLBOX_MDP_MAXIMUM_LIKELIHOOD_MODEL_HEADER_FILE

#include <tuple>
#include <random>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class models Experience as a Markov Decision Process using Maximum Likelihood.
     *
     * Often an MDP is not known in advance. It is known that it can assume
     * a certain set of states, and that a certain set of actions are
     * available to the agent, but not much more. Thus, in these cases, the
     * goal is not only to find out the best policy for the MDP we have,
     * but at the same time learn the actual transition and reward
     * functions of such a model. This task is called "reinforcement
     * learning".
     *
     * This class helps with this. A naive approach in reinforcement learning
     * is to keep track, for each action, of its results, and deduce transition
     * probabilities and rewards based on the data collected in such a way.
     * This class does just this, using Maximum Likelihood Estimates to decide
     * what the transition probabilities and rewards are.
     *
     * This class maps an Experience object to the most likely transition
     * reward functions that produced it. The transition function is guaranteed
     * to be a correct probability function, as in the sum of the probabilities
     * of all transitions from a particular state and a particular action is
     * always 1. Each instance is not directly synced with the supplied
     * Experience object. This is to avoid possible overheads, as the user can
     * optimize better depending on their use case. See sync().
     *
     * When little data is available, the deduced transition and reward
     * functions may be significantly subject to noise. A possible way to
     * improve on this is to artificially bias the data as to skew it towards
     * certain distributions.  This could be done if some knowledge of the
     * model (even approximate) is known, in order to speed up the learning
     * process. Another way is to assume that all transitions are possible, add
     * data to support that claim, and simply wait until the averages converge
     * to the true values.  Another thing that can be done is to associate with
     * each fake datapoint an high reward: this will skew the agent into trying
     * out new actions, thinking it will obtained the high rewards. This is
     * able to obtain automatically a good degree of exploration in the early
     * stages of an episode. Such a technique is called "optimistic
     * initialization".
     *
     * Whether any of these techniques work or not can definitely depend on
     * the model you are trying to approximate. Trying out things is good!
     */
    template <typename E>
    class MaximumLikelihoodModel {
        static_assert(is_experience_v<E>, "This class only works for MDP experiences!");

        public:
            using TransitionMatrix   = Matrix3D;
            using RewardMatrix       = Matrix2D;

            /**
             * @brief Constructor using previous Experience.
             *
             * This constructor selects the Experience that will
             * be used to learn an MDP Model from the data, and initializes
             * internal Model data.
             *
             * The user can choose whether he wants to directly sync the
             * MaximumLikelihoodModel to the underlying Experience, or delay it
             * for later.
             *
             * In the latter case the default transition function
             * defines a transition of probability 1 for each
             * state to itself, no matter the action.
             *
             * In general it would be better to add some amount of bias to the
             * Experience so that when a new state-action pair is tried, the
             * MaximumLikelihoodModel doesn't automatically compute 100%
             * probability of transitioning to the resulting state, but smooths
             * into it. This may depend on your problem though.
             *
             * The default reward function is 0.
             *
             * @param exp The base Experience of the model.
             * @param discount The discount used in solving methods.
             * @param sync Whether to sync with the Experience immediately or delay it.
             */
            MaximumLikelihoodModel(const E & exp, double discount = 1.0, bool sync = false);

            /**
             * @brief This function sets a new discount factor for the Model.
             *
             * @param d The new discount factor for the Model.
             */
            void setDiscount(double d);

            /**
             * @brief This function syncs the whole MaximumLikelihoodModel to the underlying Experience.
             *
             * Since use cases in AI are very varied, one may not want to
             * update its MaximumLikelihoodModel for each single transition
             * experienced by the agent. To avoid this we leave to the user the
             * task of syncing between the underlying Experience and the
             * MaximumLikelihoodModel, as he/she sees fit.
             *
             * After this function is run the transition and reward functions
             * will accurately reflect the state of the underlying Experience.
             */
            void sync();

            /**
             * @brief This function syncs a state action pair in the MaximumLikelihoodModel to the underlying Experience.
             *
             * Since use cases in AI are very varied, one may not want to
             * update the MaximumLikelihoodModel for each single transition
             * experienced by the agent. To avoid this we leave to the user the
             * task of syncing between the underlying Experience and the
             * MaximumLikelihoodModel, as he/she sees fit.
             *
             * This function updates a single state action pair with the
             * underlying Experience. This function is offered to avoid having
             * to recompute the whole MaximumLikelihoodModel if the user knows
             * that only few transitions have been experienced by the agent.
             *
             * After this function is run the transition and reward functions
             * will accurately reflect the state of the underlying Experience
             * for the specified state action pair.
             *
             * @param s The state that needs to be synced.
             * @param a The action that needs to be synced.
             */
            void sync(size_t s, size_t a);

            /**
             * @brief This function syncs a state action pair in the MaximumLikelihoodModel to the underlying Experience in the fastest possible way.
             *
             * This function updates a state action pair given that the last increased transition
             * in the underlying Experience is the triplet s, a, s1. In addition, this function only
             * works if it needs to add information from this single new point of information (if
             * more has changed from the last sync, use sync(s,a) ). The performance boost that
             * this function obtains increases with the increase of the number of states in the model.
             *
             * @param s The state that needs to be synced.
             * @param a The action that needs to be synced.
             * @param s1 The final state of the transition that got updated in the Experience.
             */
            void sync(size_t s, size_t a, size_t s1);

            /**
             * @brief This function samples the MDP for the specified state action pair.
             *
             * This function samples the model for simulate experience. The transition
             * and reward functions are used to produce, from the state action pair
             * inserted as arguments, a possible new state with respective reward.
             * The new state is picked from all possible states that the MDP allows
             * transitioning to, each with probability equal to the same probability
             * of the transition in the model. After a new state is picked, the reward
             * is the corresponding reward contained in the reward function.
             *
             * @param s The state that needs to be sampled.
             * @param a The action that needs to be sampled.
             *
             * @return A tuple containing a new state and a reward.
             */
            std::tuple<size_t, double> sampleSR(size_t s, size_t a) const;

            /**
             * @brief This function returns the number of states of the world.
             *
             * @return The total number of states.
             */
            size_t getS() const;

            /**
             * @brief This function returns the number of available actions to the agent.
             *
             * @return The total number of actions.
             */
            size_t getA() const;

            /**
             * @brief This function returns the currently set discount factor.
             *
             * @return The currently set discount factor.
             */
            double getDiscount() const;

            /**
             * @brief This function enables inspection of the underlying Experience of the MaximumLikelihoodModel.
             *
             * @return The underlying Experience of the MaximumLikelihoodModel.
             */
            const E & getExperience() const;

            /**
             * @brief This function returns the stored transition probability for the specified transition.
             *
             * @param s The initial state of the transition.
             * @param a The action performed in the transition.
             * @param s1 The final state of the transition.
             *
             * @return The probability of the specified transition.
             */
            double getTransitionProbability(size_t s, size_t a, size_t s1) const;

            /**
             * @brief This function returns the stored expected reward for the specified transition.
             *
             * @param s The initial state of the transition.
             * @param a The action performed in the transition.
             * @param s1 The final state of the transition.
             *
             * @return The expected reward of the specified transition.
             */
            double getExpectedReward(size_t s, size_t a, size_t s1) const;

            /**
             * @brief This function returns the transition matrix for inspection.
             *
             * @return The transition matrix.
             */
            const TransitionMatrix & getTransitionFunction() const;

            /**
             * @brief This function returns the transition function for a given action.
             *
             * @param a The action requested.
             *
             * @return The transition function for the input action.
             */
            const Matrix2D & getTransitionFunction(size_t a) const;

            /**
             * @brief This function returns the rewards matrix for inspection.
             *
             * @return The rewards matrix.
             */
            const RewardMatrix & getRewardFunction() const;

            /**
             * @brief This function returns whether a given state is a terminal.
             *
             * @param s The state examined.
             *
             * @return True if the input state is a terminal, false otherwise.
             */
            bool isTerminal(size_t s) const;

        private:
            size_t S, A;
            double discount_;

            const E & experience_;

            TransitionMatrix transitions_;
            RewardMatrix rewards_;

            mutable RandomEngine rand_;
    };

    template <typename E>
    MaximumLikelihoodModel<E>::MaximumLikelihoodModel(const E& exp, const double discount, const bool toSync) :
            S(exp.getS()), A(exp.getA()), experience_(exp), transitions_(A, Matrix2D(S, S)),
            rewards_(S, A), rand_(Impl::Seeder::getSeed())
    {
        setDiscount(discount);
        rewards_.setZero();

        if ( toSync ) {
            sync();
            // Sync does not touch state-action pairs which have never been
            // seen. To keep the model consistent we set all of them as
            // self-absorbing.
            for ( size_t a = 0; a < A; ++a )
                for ( size_t s = 0; s < S; ++s )
                    if ( experience_.getVisitsSum(s, a) == 0ul )
                        transitions_[a](s, s) = 1.0;
        }
        else {
            // Make transition matrix true probability
            for ( size_t a = 0; a < A; ++a )
                transitions_[a].setIdentity();
        }
    }

    template <typename E>
    void MaximumLikelihoodModel<E>::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    template <typename E>
    void MaximumLikelihoodModel<E>::sync() {
        for ( size_t a = 0; a < A; ++a )
        for ( size_t s = 0; s < S; ++s )
            sync(s,a);
    }

    template <typename E>
    void MaximumLikelihoodModel<E>::sync(const size_t s, const size_t a) {
        // Nothing to do
        const auto visitSum = experience_.getVisitsSum(s, a);
        if ( visitSum == 0ul ) return;

        // Update reward by just copying the average from experience
        rewards_(s, a) = experience_.getReward(s, a);

        // Create reciprocal for fast division
        const double visitSumReciprocal = 1.0 / visitSum;

        if constexpr (is_experience_eigen_v<E>) {
            transitions_[a].row(s) = experience_.getVisitsTable(a).row(s).template cast<double>() * visitSumReciprocal;
        } else {
            // Normalize
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                const auto visits = experience_.getVisits(s, a, s1);
                transitions_[a](s, s1) = static_cast<double>(visits) * visitSumReciprocal;
            }
        }
    }

    template <typename E>
    void MaximumLikelihoodModel<E>::sync(const size_t s, const size_t a, const size_t s1) {
        const auto visitSum = experience_.getVisitsSum(s, a);
        // The second condition is related to numerical errors. Once in a
        // while we reset those by forcing a true update using real data.
        if ( !(visitSum % 10000ul) ) return sync(s, a);

        // Update reward by just copying the average from experience
        rewards_(s, a) = experience_.getReward(s, a);

        if ( visitSum == 1ul ) {
            transitions_[a](s, s) = 0.0;
            transitions_[a](s, s1) = 1.0;
        } else {
            const double newVisits = static_cast<double>(experience_.getVisits(s, a, s1));

            const double newTransitionValue = newVisits / static_cast<double>(visitSum - 1);
            const double newVectorSum = 1.0 + (newTransitionValue - transitions_[a](s, s1));
            // This works because as long as all the values in the transition have the same denominator
            // (in this case visitSum-1), then the numerators do not matter, as we can simply normalize.
            // In the end of the process the new values will be the same as if we updated directly using
            // an increased denominator, and thus we will be able to call this function again correctly.
            transitions_[a](s, s1) = newTransitionValue;
            transitions_[a].row(s) /= newVectorSum;
        }
    }

    template <typename E>
    std::tuple<size_t, double> MaximumLikelihoodModel<E>::sampleSR(const size_t s, const size_t a) const {
        const size_t s1 = sampleProbability(S, transitions_[a].row(s), rand_);

        return std::make_tuple(s1, rewards_(s, a));
    }

    template <typename E>
    double MaximumLikelihoodModel<E>::getTransitionProbability(const size_t s, const size_t a, const size_t s1) const {
        return transitions_[a](s, s1);
    }

    template <typename E>
    double MaximumLikelihoodModel<E>::getExpectedReward(const size_t s, const size_t a, const size_t) const {
        return rewards_(s, a);
    }

    template <typename E>
    bool MaximumLikelihoodModel<E>::isTerminal(const size_t s) const {
        for ( size_t a = 0; a < A; ++a )
            if ( !checkEqualSmall(1.0, transitions_[a](s, s)) )
                return false;
        return true;
    }

    template <typename E>
    size_t MaximumLikelihoodModel<E>::getS() const { return S; }
    template <typename E>
    size_t MaximumLikelihoodModel<E>::getA() const { return A; }
    template <typename E>
    double MaximumLikelihoodModel<E>::getDiscount() const { return discount_; }
    template <typename E>
    const E & MaximumLikelihoodModel<E>::getExperience() const { return experience_; }

    template <typename E>
    const typename MaximumLikelihoodModel<E>::TransitionMatrix & MaximumLikelihoodModel<E>::getTransitionFunction() const { return transitions_; }
    template <typename E>
    const typename MaximumLikelihoodModel<E>::RewardMatrix &     MaximumLikelihoodModel<E>::getRewardFunction()     const { return rewards_; }

    template <typename E>
    const Matrix2D & MaximumLikelihoodModel<E>::getTransitionFunction(const size_t a) const { return transitions_[a]; }
}

#endif
