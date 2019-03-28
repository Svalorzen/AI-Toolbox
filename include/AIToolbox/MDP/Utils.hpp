#ifndef AI_TOOLBOX_MDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_MDP_UTILS_HEADER_FILE

#include <stddef.h>
#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This function creates and zeroes a QFunction.
     *
     * This function exists mostly to avoid remembering how to initialize
     * Eigen matrices, and does nothing special.
     *
     * @param S The state space of the QFunction.
     * @param A The action space of the QFunction.
     *
     * @return A newly built QFunction.
     */
    QFunction makeQFunction(size_t S, size_t A);

    /**
     * @brief This function creates and zeroes a ValueFunction.
     *
     * This function exists mostly to avoid remembering how to initialize
     * Eigen vectors, and does nothing special.
     *
     * @param S The state space of the ValueFunction.
     *
     * @return A newly build ValueFunction.
     */
    ValueFunction makeValueFunction(size_t S);

    /**
     * @brief This function converts a QFunction into the equivalent optimal ValueFunction.
     *
     * The ValueFunction will contain, for each state, the best action and
     * corresponding value as extracted from the input QFunction.
     *
     * @param q The QFunction to convert.
     *
     * @return The equivalent optimal ValueFunction.
     */
    ValueFunction bellmanOperator(const QFunction & q);

    /**
     * @brief This function converts a QFunction into the equivalent optimal ValueFunction.
     *
     * This function is the same as bellmanOperator(), but performs its
     * operations inplace. The input ValueFunction MUST already be sized
     * appropriately for the input QFunction.
     *
     * NOTE: This function DOES NOT perform any checks whatsoever on both
     * the validity of the input pointer and on the size of the input
     * ValueFunction. It assumes everything is already correct.
     *
     * @param q The QFunction to convert.
     * @param v A pre-allocated ValueFunction to populate with the optimal values.
     */
    void bellmanOperatorInplace(const QFunction & q, ValueFunction * v);

    /**
     * @brief This function computes all immediate rewards (state and action) of the MDP once for improved speed.
     *
     * This function pretty much creates the R(s,a) function for the input
     * model. Normally we store the reward function as R(s,a,s'), but this matrix
     * can be "compressed" into R(s,a) with no loss of meaningful information -
     * with respect to the planning process.
     *
     * Note that this function is more efficient with eigen models.
     *
     * @param m The MDP that needs to be solved.
     *
     * @return The Models's immediate rewards.
     */
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
    Matrix2D computeImmediateRewards(const M & model) {
        if constexpr(is_model_eigen_v<M>) {
            return model.getRewardFunction();
        } else {
            const auto S = model.getS();
            const auto A = model.getA();

            auto ir = QFunction(S, A);
            ir.setZero();
            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    for ( size_t s1 = 0; s1 < S; ++s1 )
                        ir(s, a) += model.getTransitionProbability(s,a,s1) * model.getExpectedReward(s,a,s1);
            return ir;
        }
    }

    /**
     * @brief This function computes the Model's QFunction from the values of a ValueFunction.
     *
     * Note that this function is more efficient with eigen models.
     *
     * @param model The MDP that needs to be solved.
     * @param v The values of the ValueFunction for the future of the QFunction.
     * @param ir The immediate rewards of the model, as created by computeImmediateRewards()
     *
     * @return A new QFunction.
     */
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
    QFunction computeQFunction(const M & model, const Values & v, QFunction ir) {
        const auto A = model.getA();

        if constexpr(is_model_eigen_v<M>) {
            for ( size_t a = 0; a < A; ++a )
                ir.col(a).noalias() += model.getTransitionFunction(a) * v;
        } else {
            const auto S = model.getS();
            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    for ( size_t s1 = 0; s1 < S; ++s1 )
                        ir(s, a) += model.getTransitionProbability(s,a,s1) * v[s1];
        }
        return ir;
    }
}

#endif
