#ifndef AI_TOOLBOX_MDP_VALUE_ITERATION_HEADER_FILE
#define AI_TOOLBOX_MDP_VALUE_ITERATION_HEADER_FILE

#include <tuple>
#include <iostream>
#include <iterator>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/Utils.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class applies the value iteration algorithm on a Model.
         *
         * This algorithm solves an MDP model for the specified horizon, or less
         * if convergence is encountered.
         *
         * The idea of this algorithm is to iteratively compute the
         * ValueFunction for the MDP optimal policy. On the first iteration,
         * the ValueFunction for horizon 1 is obtained. On the second
         * iteration, the one for horizon 2. This process is repeated until the
         * ValueFunction has converged to a specific value within a certain
         * accuracy, or the horizon requested is reached.
         *
         * This implementation in particular is ported from the MATLAB
         * MDPToolbox (although it is simplified).
         */
        class ValueIteration {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * The epsilon parameter must be >= 0.0, otherwise the
                 * constructor will throw an std::runtime_error. The epsilon
                 * parameter sets the convergence criterion. An epsilon of 0.0
                 * forces ValueIteration to perform a number of iterations
                 * equal to the horizon specified. Otherwise, ValueIteration
                 * will stop as soon as the difference between two iterations
                 * is less than the epsilon specified.
                 *
                 * Note that the default value function size needs to match
                 * the number of states of the Model. Otherwise it will
                 * be ignored. An empty value function will be defaulted
                 * to all zeroes.
                 *
                 * @param horizon The maximum number of iterations to perform.
                 * @param epsilon The epsilon factor to stop the value iteration loop.
                 * @param v The initial value function from which to start the loop.
                 */
                ValueIteration(unsigned horizon, double epsilon = 0.001, ValueFunction v = ValueFunction(Values(0), Actions(0)));

                /**
                 * @brief This function applies value iteration on an MDP to solve it.
                 *
                 * The algorithm is constrained by the currently set parameters.
                 *
                 * @tparam M The type of the solvable MDP.
                 * @param m The MDP that needs to be solved.
                 * @return A tuple containing a boolean value specifying whether
                 *         the specified epsilon bound was reached and the
                 *         ValueFunction and the QFunction for the Model.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                std::tuple<bool, ValueFunction, QFunction> operator()(const M & m);

                /**
                 * @brief This function sets the epsilon parameter.
                 *
                 * The epsilon parameter must be >= 0.0, otherwise the
                 * constructor will throw an std::runtime_error. The epsilon
                 * parameter sets the convergence criterion. An epsilon of 0.0
                 * forces ValueIteration to perform a number of iterations
                 * equal to the horizon specified. Otherwise, ValueIteration
                 * will stop as soon as the difference between two iterations
                 * is less than the epsilon specified.
                 *
                 * @param e The new epsilon parameter.
                 */
                void setEpsilon(double e);

                /**
                 * @brief This function sets the horizon parameter.
                 *
                 * @param h The new horizon parameter.
                 */
                void setHorizon(unsigned h);

                /**
                 * @brief This function sets the starting value function.
                 *
                 * An empty value function defaults to all zeroes. Note
                 * that the default value function size needs to match
                 * the number of states of the Model that needs to be
                 * solved. Otherwise it will be ignored.
                 *
                 * @param v The new starting value function.
                 */
                void setValueFunction(ValueFunction v);

                /**
                 * @brief This function will return the currently set epsilon parameter.
                 *
                 * @return The currently set epsilon parameter.
                 */
                double getEpsilon() const;

                /**
                 * @brief This function will return the current horizon parameter.
                 *
                 * @return The currently set horizon parameter.
                 */
                unsigned getHorizon() const;

                /**
                 * @brief This function will return the current set default value function.
                 *
                 * @return The currently set default value function.
                 */
                const ValueFunction & getValueFunction() const;

            private:
                // Parameters
                double discount_, epsilon_;
                unsigned horizon_;
                ValueFunction vParameter_;

                // Internals
                ValueFunction v1_;
                size_t S, A;

                // Internal methods
                /**
                 * @brief This function computes all immediate rewards (state and action) of the MDP once for improved speed.
                 *
                 * @tparam M The type of the solvable MDP.
                 * @param m The MDP that needs to be solved.
                 *
                 * @return The Models's immediate rewards.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                Table2D computeImmediateRewards(const M & model) const;

                /**
                 * @brief This function creates the Model's most up-to-date QFunction.
                 *
                 * @tparam M The type of the solvable MDP.
                 *
                 * @param m The MDP that needs to be solved.
                 * @param ir The immediate rewards of the model.
                 *
                 * @return A new QFunction.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                QFunction computeQFunction(const M & model, const Table2D & ir) const;

                /**
                 * @brief This function applies a single pass Bellman operator, improving the current ValueFunction estimate.
                 *
                 * This function computes the optimal value and action for
                 * each state, given the precomputed QFunction.
                 *
                 * @param q The precomputed QFunction.
                 * @param vOut The newly estimated ValueFunction.
                 */
                inline void bellmanOperator(const QFunction & q, ValueFunction * vOut) const;
        };

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        std::tuple<bool, ValueFunction, QFunction> ValueIteration::operator()(const M & model) {
            // Extract necessary knowledge from model so we don't have to pass it around
            S = model.getS();
            A = model.getA();
            discount_ = model.getDiscount();

            {
                // Verify that parameter value function is compatible.
                size_t size = std::get<VALUES>(vParameter_).size();
                if ( size != S ) {
                    if ( size != 0 )
                        std::cerr << "AIToolbox: Size of starting value function in ValueIteration::solve() is incorrect, ignoring...\n";
                    // Defaulting
                    v1_ = makeValueFunction(S);
                }
                else
                    v1_ = vParameter_;
            }

            auto ir = computeImmediateRewards(model);

            unsigned timestep = 0;
            double variation = epsilon_ * 2; // Make it bigger

            Values val0;
            QFunction q = makeQFunction(S, A);

            bool useEpsilon = checkDifferentSmall(epsilon_, 0.0);
            while ( timestep < horizon_ && (!useEpsilon || variation > epsilon_) ) {
                ++timestep;

                auto & val1 = std::get<VALUES>(v1_);
                val0 = val1;

                q = computeQFunction(model, ir);
                bellmanOperator(q, &v1_);

                // We do this only if the epsilon specified is positive, otherwise we
                // continue for all the timesteps.
                if ( useEpsilon ) {
                    auto computeVariation = [](double lhs, double rhs) { return std::fabs(lhs - rhs); };
                    // We compute the difference and store it into v0 for comparison.
                    std::transform(std::begin(val1), std::end(val1), std::begin(val0), std::begin(val0), computeVariation);

                    variation = *std::max_element(std::begin(val0), std::end(val0));
                }
            }

            // We do not guarantee that the Value/QFunctions are the perfect ones, as we stop as within epsilon.
            return std::make_tuple(variation <= epsilon_, v1_, q);
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        Table2D ValueIteration::computeImmediateRewards(const M & model) const {
            Table2D pr(boost::extents[S][A]);

            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    for ( size_t s1 = 0; s1 < S; ++s1 )
                        pr[s][a] += model.getTransitionProbability(s,a,s1) * model.getExpectedReward(s,a,s1);

            return pr;
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        QFunction ValueIteration::computeQFunction(const M & model, const Table2D & ir) const {
            QFunction q = ir;

            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    for ( size_t s1 = 0; s1 < S; ++s1 )
                        q[s][a] += model.getTransitionProbability(s,a,s1) * discount_ * std::get<VALUES>(v1_)[s1];
            return q;
        }

        void ValueIteration::bellmanOperator(const QFunction & q, ValueFunction * v) const {
            auto & values  = std::get<VALUES> (*v);
            auto & actions = std::get<ACTIONS>(*v);

            for ( size_t s = 0; s < S; ++s ) {
                // Accessing an element like this creates a temporary. Thus we need to bind it.
                QFunction::const_reference ref = q[s];
                auto begin = std::begin(ref);
                auto it = std::max_element(begin, std::end(ref));

                values[s] = *it;
                actions[s] = std::distance(begin, it);
            }
        }
    }
}

#endif
