#ifndef AI_TOOLBOX_MDP_THOMPSON_MODEL_HEADER_FILE
#define AI_TOOLBOX_MDP_THOMPSON_MODEL_HEADER_FILE

#include <tuple>
#include <random>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class models Experience as a Markov Decision Process using Thompson Sampling.
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
     * This class does just this, using Thompson Sampling to decide what the
     * transition probabilities and rewards are.
     *
     * This class maps an Experience object using a series of Dirichlet (for
     * transitions) and Student-t (for rewards) distributions, one per
     * state-action pairs. The user can sample from these distributions to
     * obtain transition and reward functions. As more data is accumulated, the
     * distributions can be resampled so that these functions better reflect
     * the data. The syncing operation MUST be done manually as it is slightly
     * expensive (it must sample a distribution with S parameters and normalize
     * the result). See sync().
     *
     * When little data is available, syncing will generally result in
     * transition functions where most transitions are assumed possible. Priors
     * can be given to the Experience as "fictional" experience so as to bias
     * the result. Additionally, this class uses Jeffreys prior when sampling.
     * For a Dirichlet distribution, this is equivalent to having 0.5 priors on
     * all parameters (which can't be set via the Experience, as they are not
     * integers). For the reward, the posteriors are student-t distributions. A
     * Jeffreys prior ensures that the sampling is non-biased through any
     * transformation of the original parameters.
     *
     * The strength of ThompsonModel is that it can replace traditional
     * exploration techniques, embedding our beliefs of what transitions and
     * rewards are possible directly in the sampled functions.
     *
     * Whether any of these techniques work or not can definitely depend on
     * the model you are trying to approximate. Trying out things is good!
     */
    template <typename E>
    class ThompsonModel {
        static_assert(is_experience_v<E>, "This class only works for MDP experiences!");

        public:
            using TransitionMatrix   = Matrix3D;
            using RewardMatrix       = Matrix2D;

            /**
             * @brief Constructor using previous Experience.
             *
             * This constructor selects the Experience that will be used to
             * learn an MDP Model from the data, and initializes internal Model
             * data.
             *
             * Differently from MaximumLikelihoodModel, we always sync at
             * first, since we will sample from a Dirichlet distribution
             * whether we have data or not.
             *
             * All transition parameters read from the Experience will be
             * incremented by 0.5, since we are using Jeffreys prior.
             *
             * The rewards will be sampled from student-t distributions.
             *
             * @param exp The base Experience of the model.
             * @param discount The discount used in solving methods.
             */
            ThompsonModel(const E & exp, double discount = 1.0);

            /**
             * @brief This function sets a new discount factor for the Model.
             *
             * @param d The new discount factor for the Model.
             */
            void setDiscount(double d);

            /**
             * @brief This function syncs the whole ThompsonModel to the underlying Experience.
             *
             * Since use cases in AI are very varied, one may not want to
             * update its ThompsonModel for each single transition experienced
             * by the agent. To avoid this we leave to the user the task of
             * syncing between the underlying Experience and the ThompsonModel,
             * as he/she sees fit.
             */
            void sync();

            /**
             * @brief This function syncs a state action pair in the ThompsonModel to the underlying Experience.
             *
             * Since use cases in AI are very varied, one may not want to
             * update the ThompsonModel for each single transition
             * experienced by the agent. To avoid this we leave to the user the
             * task of syncing between the underlying Experience and the
             * ThompsonModel, as he/she sees fit.
             *
             * This function updates a single state action pair with the
             * underlying Experience. This function provides a higher
             * fine-grained control on resampling than sync().
             *
             * Both transitions and rewards for the specified state-action pair
             * will be resampled.
             *
             * @param s The state that needs to be synced.
             * @param a The action that needs to be synced.
             */
            void sync(size_t s, size_t a);

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
             * @brief This function enables inspection of the underlying Experience of the ThompsonModel.
             *
             * @return The underlying Experience of the ThompsonModel.
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
    ThompsonModel<E>::ThompsonModel(const E& exp, const double discount) :
            S(exp.getS()), A(exp.getA()), experience_(exp), transitions_(A, Matrix2D(S, S)),
            rewards_(S, A), rand_(Impl::Seeder::getSeed())
    {
        setDiscount(discount);

        sync();
    }

    template <typename E>
    void ThompsonModel<E>::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    template <typename E>
    void ThompsonModel<E>::sync() {
        for ( size_t a = 0; a < A; ++a )
        for ( size_t s = 0; s < S; ++s )
            sync(s,a);
    }

    template <typename E>
    void ThompsonModel<E>::sync(const size_t s, const size_t a) {
        if constexpr (is_experience_eigen_v<E>) {
            sampleDirichletDistribution(
                // Here we add the Jeffreys prior
                //
                // Ideally this shouldn't allocate, as the casting and sum
                // should simply create a wrapper Eigen object which is passed
                // by reference, so should be still as efficient as doing it by
                // hand.
                experience_.getVisitsTable(a).row(s).array().template cast<double>() + 0.5,
                rand_, transitions_[a].row(s)
            );
        } else {
            // Sample manually
            double sum = 0.0;
            for (size_t s1 = 0; s1 < S; ++s1) {
                // Here we add the Jeffreys prior
                std::gamma_distribution<double> dist(experience_.getVisits(s, a, s1) + 0.5, 1.0);
                transitions_[a](s, s1) = dist(rand_);
                sum += transitions_[a](s, s1);
            }
            transitions_[a].row(s) /= sum;
        }

        const auto visits = experience_.getVisitsSum(s, a);
        const auto MLEReward = experience_.getReward(s, a);
        const auto M2 = experience_.getM2(s, a);
        if (visits < 2) {
            // If we don't have enough info for the STD, we revert to MLE.
            rewards_(s, a) = MLEReward;
        } else {
            std::student_t_distribution<double> dist(visits - 1);
            rewards_(s, a) = MLEReward + dist(rand_) * std::sqrt(M2 / (visits * (visits - 1)));
        }
    }

    template <typename E>
    std::tuple<size_t, double> ThompsonModel<E>::sampleSR(const size_t s, const size_t a) const {
        const size_t s1 = sampleProbability(S, transitions_[a].row(s), rand_);

        return std::make_tuple(s1, rewards_(s, a));
    }

    template <typename E>
    double ThompsonModel<E>::getTransitionProbability(const size_t s, const size_t a, const size_t s1) const {
        return transitions_[a](s, s1);
    }

    template <typename E>
    double ThompsonModel<E>::getExpectedReward(const size_t s, const size_t a, const size_t) const {
        return rewards_(s, a);
    }

    template <typename E>
    bool ThompsonModel<E>::isTerminal(const size_t s) const {
        for ( size_t a = 0; a < A; ++a )
            if ( !checkEqualSmall(1.0, transitions_[a](s, s)) )
                return false;
        return true;
    }

    template <typename E>
    size_t ThompsonModel<E>::getS() const { return S; }
    template <typename E>
    size_t ThompsonModel<E>::getA() const { return A; }
    template <typename E>
    double ThompsonModel<E>::getDiscount() const { return discount_; }
    template <typename E>
    const E & ThompsonModel<E>::getExperience() const { return experience_; }

    template <typename E>
    const typename ThompsonModel<E>::TransitionMatrix & ThompsonModel<E>::getTransitionFunction() const { return transitions_; }
    template <typename E>
    const typename ThompsonModel<E>::RewardMatrix &     ThompsonModel<E>::getRewardFunction()     const { return rewards_; }

    template <typename E>
    const Matrix2D & ThompsonModel<E>::getTransitionFunction(const size_t a) const { return transitions_[a]; }
}

#endif
