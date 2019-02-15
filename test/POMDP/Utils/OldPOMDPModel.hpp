#ifndef AI_TOOLBOX_OLD_POMDP_MODEL_HEADER_FILE
#define AI_TOOLBOX_OLD_POMDP_MODEL_HEADER_FILE

#include <random>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Types.hpp>
#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = std::enable_if_t<AIToolbox::MDP::is_model_v<M>>>
        class OldPOMDPModel;
#endif

/**
 * @brief This class represents a Partially Observable Markov Decision Process.
 *
 * This class inherits from any valid MDP model type, so that it can
 * use its base methods, and it builds from those. Templated inheritance
 * was chosen to improve performance and keep code small, instead of
 * doing composition.
 *
 * A POMDP is an MDP where the agent, at each timestep, does not know
 * in which state it is. Instead, after each action is performed, it
 * obtains an "observation", which offers some information as to which
 * new state the agent has transitioned to. This observation is
 * determined by an "observation function", that maps S'xAxO to a
 * probability: the probability of obtaining observation O after taking
 * action A and *landing* in state S'.
 *
 * Since now its knowledge is imperfect, in order to represent the
 * knowledge of the state it is currently in, the agent is thus forced
 * to use Beliefs: probability distributions over states.
 *
 * The way a Belief works is that, after each action and observation,
 * the agent can reason as follows: given my previous Belief
 * (distribution over states) that I think I was in, what is now the
 * probability that I transitioned to any particular state? This new
 * Belief can be computed from the Model, given that the agent knows
 * the distributions of the transition and observation functions.
 *
 * Turns out that a POMDP can be viewed as an MDP with an infinite
 * number of states, where each state is essentially a Belief. Since a
 * Belief is a vector of real numbers, there are infinite of them, thus
 * the infinite number of states. While POMDPs can be much more
 * powerful than MDPs for modeling real world problems, where
 * information is usually not perfect, it turns out that this
 * infinite-state property makes them so much harder to solve
 * perfectly, and their solutions much more complex.
 *
 * A POMDP solution is composed by several policies, which apply in
 * different ranges of the Belief space, and suggest different actions
 * depending on the observations received by the agent at each
 * timestep. The values of those policies can be, in the same way,
 * represented as a number of value vectors (called alpha vectors in
 * the literature) that apply in those same ranges of the Belief space.
 * Each alpha vector is somewhat similar to an MDP ValueFunction.
 *
 * @tparam M The particular MDP type that we want to extend.
 */
template <typename M>
class OldPOMDPModel<M> : public M {
    public:
        using ObservationMatrix = AIToolbox::DumbMatrix3D;

        /**
         * @brief Basic constructor.
         *
         * This constructor initializes the observation function
         * so that all actions will return observation 0.
         *
         * @tparam Args All types of the parent constructor arguments.
         * @param o The number of possible observations the agent could make.
         * @param parameters All arguments needed to build the parent Model.
         */
        template <typename... Args>
        OldPOMDPModel(size_t o, Args&&... parameters);

        /**
         * @brief Basic constructor.
         *
         * This constructor takes an arbitrary three dimensional
         * containers and tries to copy its contents into the
         * observations matrix.
         *
         * The container needs to support data access through
         * operator[]. In addition, the dimensions of the
         * container must match the ones provided as arguments
         * both directly (o) and indirectly (s,a).
         *
         * This is important, as this constructor DOES NOT perform
         * any size checks on the external containers.
         *
         * Internal values of the containers will be converted to double,
         * so these conversions must be possible.
         *
         * In addition, the observation container must contain a
         * valid transition function.
         * \sa transitionCheck()
         *
         * \sa copyDumb3D()
         *
         * @tparam ObFun The external observations container type.
         * @param o The number of possible observations the agent could make.
         * @param of The observation probability matrix.
         * @param parameters All arguments needed to build the parent Model.
         */
        // Check that ObFun is a triple-matrix, otherwise we'll call the other constructor!
        template <typename ObFun, typename... Args, typename = std::enable_if_t<std::is_constructible_v<double,decltype(std::declval<ObFun>()[0][0][0])>>>
        OldPOMDPModel(size_t o, ObFun && of, Args&&... parameters);

        /**
         * @brief Copy constructor from any valid POMDP model.
         *
         * This allows to copy from any other model. A nice use for this is to
         * convert any model which computes probabilities on the fly into an
         * POMDP::Model where probabilities are all stored for fast access. Of
         * course such a solution can be done only when the number of states,
         * actions and observations is not too big.
         *
         * Of course this constructor is available only if the underlying Model
         * allows to be constructed too.
         *
         * @tparam PM The type of the other model.
         * @param model The model that needs to be copied.
         */
        template <typename PM, typename = std::enable_if_t<AIToolbox::POMDP::is_model_v<PM> && std::is_constructible_v<M,PM>>>
        OldPOMDPModel(const PM& model);

        /**
         * @brief This function replaces the Model observation function with the one provided.
         *
         * The container needs to support data access through
         * operator[]. In addition, the dimensions of the
         * containers must match the ones provided as arguments
         * (for three dimensions: s,a,o).
         *
         * This is important, as this constructor DOES NOT perform
         * any size checks on the external containers.
         *
         * Internal values of the container will be converted to double,
         * so these conversions must be possible.
         *
         * @tparam ObFun The external observations container type.
         * @param of The external observations container.
         */
        template <typename ObFun>
        void setObservationFunction(const ObFun & of);

        /**
         * @brief This function samples the POMDP for the specified state action pair.
         *
         * This function samples the model for simulated experience. The
         * transition, observation and reward functions are used to
         * produce, from the state action pair inserted as arguments, a
         * possible new state with respective observation and reward.
         * The new state is picked from all possible states that the
         * MDP allows transitioning to, each with probability equal to
         * the same probability of the transition in the model. After a
         * new state is picked, an observation is sampled from the
         * observation function distribution, and finally the reward is
         * the corresponding reward contained in the reward function.
         *
         * @param s The state that needs to be sampled.
         * @param a The action that needs to be sampled.
         *
         * @return A tuple containing a new state, observation and reward.
         */
        std::tuple<size_t,size_t, double> sampleSOR(size_t s,size_t a) const;

        /**
         * @brief This function samples the POMDP for the specified state action pair.
         *
         * This function samples the model for simulated experience.
         * The transition, observation and reward functions are used to
         * produce, from the state, action and new state inserted as
         * arguments, a possible new observation and reward. The
         * observation and rewards are picked so that they are
         * consistent with the specified new state.
         *
         * @param s The state that needs to be sampled.
         * @param a The action that needs to be sampled.
         * @param s1 The resulting state of the s,a transition.
         *
         * @return A tuple containing a new observation and reward.
         */
        std::tuple<size_t, double> sampleOR(size_t s,size_t a,size_t s1) const;

        /**
         * @brief This function returns the stored observation probability for the specified state-action pair.
         *
         * @param s1 The final state of the transition.
         * @param a The action performed in the transition.
         * @param o The recorded observation for the transition.
         *
         * @return The probability of the specified observation.
         */
        double getObservationProbability(size_t s1, size_t a, size_t o) const;

        /**
         * @brief This function returns the number of observations possible.
         *
         * @return The total number of observations.
         */
        size_t getO() const;

        /**
         * @brief This function returns the observation matrix for inspection.
         *
         * @return The observation matrix.
         */
        const ObservationMatrix & getObservationFunction() const;

    private:
        size_t O;
        ObservationMatrix observations_;
        // We need this because we don't know if our parent already has one,
        // and we wouldn't know how to access it!
        mutable AIToolbox::RandomEngine rand_;
};

template <typename M>
template <typename... Args>
OldPOMDPModel<M>::OldPOMDPModel(size_t o, Args&&... params) : M(std::forward<Args>(params)...), O(o), observations_(boost::extents[this->getS()][this->getA()][O]),
                                              rand_(AIToolbox::Impl::Seeder::getSeed())
{
    for ( size_t s = 0; s < this->getS(); ++s )
        for ( size_t a = 0; a < this->getA(); ++a )
            observations_[s][a][0] = 1.0;
}

template <typename M>
template <typename ObFun, typename... Args, typename>
OldPOMDPModel<M>::OldPOMDPModel(size_t o, ObFun && of, Args&&... params) : M(std::forward<Args>(params)...), O(o), observations_(boost::extents[this->getS()][this->getA()][O]),
                                                                rand_(AIToolbox::Impl::Seeder::getSeed())
{
    setObservationFunction(of);
}

template <typename M>
template <typename PM, typename>
OldPOMDPModel<M>::OldPOMDPModel(const PM& model) : M(model), O(model.getO()), observations_(boost::extents[this->getS()][this->getA()][O]),
                                   rand_(AIToolbox::Impl::Seeder::getSeed())
{
    for ( size_t s1 = 0; s1 < this->getS(); ++s1 )
        for ( size_t a = 0; a < this->getA(); ++a ) {
            for ( size_t o = 0; o < O; ++o ) {
                observations_[s1][a][o] = model.getObservationProbability(s1, a, o);
            }
            if ( ! AIToolbox::isProbability(O, observations_[s1][a]) ) throw std::invalid_argument("Input observation matrix does not contain valid probabilities.");
        }
}

template <typename M>
template <typename ObFun>
void OldPOMDPModel<M>::setObservationFunction(const ObFun & of) {
    for ( size_t s1 = 0; s1 < this->getS(); ++s1 )
        for ( size_t a = 0; a < this->getA(); ++a )
            if ( ! AIToolbox::isProbability(O, of[s1][a]) ) throw std::invalid_argument("Input observation matrix does not contain valid probabilities.");

    copyDumb3D(of, observations_, this->getS(), this->getA(), O);
}

template <typename M>
double OldPOMDPModel<M>::getObservationProbability(size_t s1, size_t a, size_t o) const {
    return observations_[s1][a][o];
}

template <typename M>
size_t OldPOMDPModel<M>::getO() const {
    return O;
}

template <typename M>
const typename OldPOMDPModel<M>::ObservationMatrix & OldPOMDPModel<M>::getObservationFunction() const {
    return observations_;
}

template <typename M>
std::tuple<size_t,size_t, double> OldPOMDPModel<M>::sampleSOR(size_t s, size_t a) const {
    size_t s1, o;
    double r;

    std::tie(s1, r) = this->sampleSR(s, a);
    o = AIToolbox::sampleProbability(O, observations_[s1][a], rand_);

    return std::make_tuple(s1, o, r);
}

template <typename M>
std::tuple<size_t, double> OldPOMDPModel<M>::sampleOR(size_t s, size_t a, size_t s1) const {
    size_t o = AIToolbox::sampleProbability(O, observations_[s1][a], rand_);
    double r = this->getExpectedReward(s, a, s1);
    return std::make_tuple(o, r);
}

#endif
