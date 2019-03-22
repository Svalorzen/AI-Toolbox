#ifndef AI_TOOLBOX_MDP_LINEAR_PROGRAMMING_HEADER_FILE
#define AI_TOOLBOX_MDP_LINEAR_PROGRAMMING_HEADER_FILE

#include <AIToolbox/Utils/LP.hpp>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class solves an MDP using Linear Programming.
     *
     * This class is a very simple wrapper for solving an MDP using linear
     * programming. The solution can only be computed for infinite horizons,
     * and the precision is the ones used by the underlying LP library.
     *
     * It creates a set of |S| variables and |S|*|A| constraints, which when
     * solved obtain the optimal ValueFunction values.
     *
     * From there we compute the optimal QFunction, and we return them.
     */
    class LinearProgramming {
        public:
            /**
             * @brief This function solves the input MDP using linear programming.
             *
             * @tparam M The type of the solvable MDP.
             * @param m The MDP that needs to be solved.
             *
             * @return A tuple containing the maximum variation for the
             *         ValueFunction, the ValueFunction and the QFunction for
             *         the Model.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            std::tuple<double, ValueFunction, QFunction> operator()(const M & m);
    };

    template <typename M, typename>
    std::tuple<double, ValueFunction, QFunction> LinearProgramming::operator()(const M & model) {
        // Extract necessary knowledge from model so we don't have to pass it around
        const size_t S = model.getS();
        const size_t A = model.getA();

        // Here we solve an LP to determine the optimal value function for the
        // infinite horizon. In particular, for every state, we represent its
        // value with a variable (we assume an uniform distribution over the
        // states here).
        //
        // Then we minimize the sum of the variables, subject to:
        //
        //     sum_s' T(s,a,s') * [ R(s,a,s') + gamma * V*(s') ] <= V(s)
        //
        // for every combination of s and a (so |S|*|A| constraints in the
        // end).
        //
        // Here we transform the constraints in the form:
        //
        //     V(s) - sum_s' gamma * T(s,a,s') * V*(s') >= sum_s' T(s,a,s') * R(s,a,s')
        //
        // and we merge the V(s) with its appropriate V*(s') element.
        LP lp(S);
        lp.resize(S * A);

        // Assume uniform distribution, and minimize the objective.
        lp.row.fill(1.0 / S);
        lp.setObjective(false);

        for (size_t s = 0; s < S; ++s) {
            // For every variable, we set it as unbounded (as its value can be
            // anything).
            lp.setUnbounded(s);
            for (size_t a = 0; a < A; ++a) {
                double rhs;
                if constexpr(is_model_eigen_v<M>) {
                    lp.row = -model.getDiscount() * model.getTransitionFunction(a).row(s);
                    rhs = model.getRewardFunction()(s, a);
                } else {
                    // For each constraint, we compute the RHS, while at the same
                    // time setting the coefficients for the various variables.
                    rhs = 0.0;
                    for (size_t s1 = 0; s1 < S; ++s1) {
                        lp.row[s1] = -model.getDiscount() * model.getTransitionProbability(s, a, s1);
                        rhs += model.getTransitionProbability(s, a, s1) * model.getExpectedReward(s, a, s1);
                    }
                }
                // Finally we add the V(s) at its place.
                lp.row[s] += 1.0;
                lp.pushRow(LP::Constraint::GreaterEqual, rhs);
            }
        }

        // We solve the LP, and get V*
        auto values = lp.solve(S);

        if (!values)
            throw std::runtime_error("Could not solve the LP for this MDP");

        // We have the values, but we also want the optimal actions. So while
        // we're at it, we also build Q.
        const auto & ir = [&]{
            if constexpr (is_model_eigen_v<M>) return model.getRewardFunction();
            else return computeImmediateRewards(model);
        }();

        auto q = computeQFunction(model, model.getDiscount() * (*values), ir);

        ValueFunction v;
        v.values = std::move(*values);
        v.actions.resize(S);
        for (size_t s = 0; s < S; ++s)
            q.row(s).maxCoeff(&v.actions[s]);

        return std::make_tuple(lp.getPrecision(), std::move(v), std::move(q));
    }
}

#endif
