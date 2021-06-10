#ifndef AI_TOOLBOX_MDP_TYPE_TRAITS_HEADER_FILE
#define AI_TOOLBOX_MDP_TYPE_TRAITS_HEADER_FILE

#include <AIToolbox/TypeTraits.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This concept represents the required interface for a generative MDP.
     *
     * This concept tests for the interface of a generative MDP model.
     * The interface must be implemented and be public in the parameter
     * class. The interface is the following:
     *
     * - size_t getS() const : Returns the number of states of the Model.
     * - size_t getA() const : Returns the number of actions of the Model.
     * - double getDiscount() const : Returns the discount factor of the Model.
     * - std::tuple<size_t, double> sampleSR(size_t s, size_t a) const : Returns a sampled state-reward pair from (s,a)
     * - bool isTerminal(size_t s) const : Reports whether the input state is a terminal state.
     *
     * The concept re-uses the "base" concept and simply requires a fixed
     * action space, and integral state and action spaces.
     *
     * \sa AIToolbox::IsGenerativeModel
     */
    template <typename M>
    concept IsGenerativeModel = AIToolbox::IsGenerativeModel<M> &&
                                HasIntegralStateSpace<M> &&
                                HasIntegralActionSpace<M> &&
                                HasFixedActionSpace<M>;

    /**
     * @brief This concept represents the required interface for a full MDP model.
     *
     * This concept tests for the interface of an MDP model.
     * The interface must be implemented and be public in the parameter
     * class. The interface is the following:
     *
     * - double getTransitionProbability(size_t s, size_t a, size_t s1) const : Returns the transition probability given (s,a) to s1
     * - double getExpectedReward(size_t s, size_t a, size_t s1) const : Returns the expected reward for transition (s,a) to s1
     *
     * In addition the MDP needs to respect the interface for the MDP generative model.
     *
     * \sa IsGenerativeModel
     */
    template <typename M>
    concept IsModel = IsGenerativeModel<M> && requires (const M m, size_t s, size_t a) {
        { m.getTransitionProbability(s, a, s) } -> std::convertible_to<double>;
        { m.getExpectedReward(s, a, s) }        -> std::convertible_to<double>;
    };

    /**
     * @brief This concept represents the required interface that allows MDP algorithms to leverage Eigen.
     *
     * This struct tests for the interface of an MDP model which uses Eigen
     * matrices internally. This should work for both dense and sparse models.
     *
     * The interface must be implemented and be public in the parameter
     * class. The interface is the following:
     *
     * - T getTransitionFunction(size_t a) const : Returns the transition function for a given action as a matrix SxS', where T is some Eigen matrix type.
     * - R getRewardFunction() const : Returns the reward function as a matrix SxA', where R is some Eigen matrix type.
     *
     * In addition the MDP needs to respect the interface for the MDP model.
     *
     * \sa IsModel
     */
    template <typename M>
    concept IsModelEigen = IsModel<M> && requires (const M m, size_t a) {
        m.getTransitionFunction(a);
        requires IsDerivedFromEigen<std::remove_cvref_t<decltype((m.getTransitionFunction(a)))>>;

        m.getRewardFunction();
        requires IsDerivedFromEigen<std::remove_cvref_t<decltype((m.getRewardFunction()))>>;
    };

    /**
     * @brief This concept represents the required interface for an experience recorder.
     *
     * This concept tests for the interface of an experience recorder that can
     * be used to create Reinforcement Learning MDP models.
     *
     * The interface must be implemented and be public in the parameter class.
     * The interface is the following:
     *
     * - long unsigned getVisits(size_t, size_t, size_t) const : Returns the number of times a particular transition has been experienced.
     * - long unsigned getVisitsSum(size_t, size_t) const : Returns the number of times a transition starting with the input state-action pair.
     * - double getReward(size_t, size_t) const : Returns the expected rewards obtained from a given state-action pair.
     * - double getM2(size_t, size_t) const : Returns the M2 statistics of experienced rewards from the given state-action pair.
     */
    template <typename E>
    concept IsExperience = requires (const E e, size_t s, size_t a) {
        { e.getVisits(s, a, s) } -> std::convertible_to<long unsigned>;
        { e.getVisitsSum(s, a) } -> std::convertible_to<long unsigned>;
        { e.getReward(s, a) }    -> std::convertible_to<double>;
        { e.getM2(s, a) }        -> std::convertible_to<double>;
    };

    /**
     * @brief This concept represents the required Experience interface that allows leverage Eigen.
     *
     * This concept tests for the interface of an MDP Experience which uses
     * Eigen matrices internally. This should work for both dense and sparse
     * Experiences.
     *
     * The interface must be implemented and be public in the parameter
     * class. The interface is the following:
     *
     * - T getVisitsTable(size_t a) const : Returns the visits table for a given action as a matrix SxS', where T is some Eigen matrix type.
     * - R getRewardFunction() const : Returns the reward matrix as a matrix SxA', where R is some Eigen matrix type.
     * - S getM2Matrix() const : Returns the M2 matrix as a matrix SxA', where S is some Eigen matrix type.
     *
     * In addition the Experience needs to respect the basic Experience interface.
     *
     * \sa IsExperience
     */
    template <typename E>
    concept IsExperienceEigen = IsExperience<E> && requires (const E e, size_t a) {
        e.getVisitsSumTable();
        requires IsDerivedFromEigen<std::remove_cvref_t<decltype((e.getVisitsSumTable()))>>;

        e.getVisitsTable(a);
        requires IsDerivedFromEigen<std::remove_cvref_t<decltype((e.getVisitsTable(a)))>>;

        e.getRewardMatrix();
        requires IsDerivedFromEigen<std::remove_cvref_t<decltype((e.getRewardMatrix()))>>;

        e.getM2Matrix();
        requires IsDerivedFromEigen<std::remove_cvref_t<decltype((e.getM2Matrix()))>>;
    };
}

#endif
