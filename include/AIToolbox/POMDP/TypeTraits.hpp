#ifndef AI_TOOLBOX_POMDP_TYPE_TRAITS_HEADER_FILE
#define AI_TOOLBOX_POMDP_TYPE_TRAITS_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This concept represents the required interface for a generative POMDP.
     *
     * This concept tests for the interface of a generative POMDP model. The
     * interface must be implemented and be public in the parameter class. The
     * interface is the following:
     *
     * - std::tuple<size_t, size_t, double> sampleSOR(size_t s, size_t a) const : Returns a sampled state-observation-reward tuple from (s,a)
     *
     * In addition the POMDP needs to respect the interface for the MDP generative model.
     *
     * \sa MDP::IsGenerativeModel
     *
     * Note that, at least for now, we can avoid asking this generative
     * model for the total number of observation possible, because they are
     * not required as parameters for the functions, but just returned.
     * This may change in future though, depending on algorithms'
     * requirements.
     *
     */
    template <typename M>
    concept IsGenerativeModel = MDP::IsGenerativeModel<M> && requires (M m) {
        { m.sampleSOR(m.getS(), m.getA()) } -> std::convertible_to<std::tuple<size_t, size_t, double>>;
    };

    /**
     * @brief This concept represents the required interface for a POMDP Model.
     *
     * This struct tests for the interface of a POMDP model.
     *
     * The interface must be implemented and be public in the parameter class.
     * The interface is the following:
     *
     * - size_t getO() const : Returns the number of observations of the Model.
     * - double getObservationProbability(size_t s1, size_t a, size_t o) const : Returns the probability for observation o after action a and final state s1.
     *
     * In addition the POMDP needs to respect the interface for the POMDP generative
     * model and the MDP model.
     *
     * \sa IsGenerativeModel
     * \sa MDP::IsModel
     * \sa HasIntegralObservationSpace
     */
    template <typename M>
    concept IsModel = MDP::IsModel<M> && IsGenerativeModel<M> && HasIntegralObservationSpace<M> && requires (M m, size_t s, size_t a) {
        { m.getObservationProbability(s, a, s) } -> std::convertible_to<double>;
    };

    /**
     * @brief This concept represents the required interface that allows POMDP algorithms to leverage Eigen.
     *
     * This concept tests for the interface of a POMDP model which uses Eigen
     * matrices internally.
     *
     * The interface must be implemented and be public in the parameter class.
     * The interface is the following:
     *
     * - O getObservationFunction(size_t a) const : Returns the observation function for a given action as a matrix S'xO, where O is some Eigen matrix type.
     *
     * In addition the POMDP needs to respect the interface for the POMDP model
     * and the Eigen MDP model.
     *
     * \sa POMDP::IsModel
     * \sa MDP::IsModelEigen
     */
    template <typename M>
    concept IsModelEigen = MDP::IsModelEigen<M> && IsModel<M> && requires (M m, size_t a) {
        m.getObservationFunction(a);
        requires IsDerivedFromEigen<std::remove_cvref_t<decltype((m.getObservationFunction(a)))>>;
    };
}

#endif
