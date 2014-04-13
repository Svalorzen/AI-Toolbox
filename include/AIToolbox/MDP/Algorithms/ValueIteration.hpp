#ifndef AI_TOOLBOX_MDP_VALUE_ITERATION_HEADER_FILE
#define AI_TOOLBOX_MDP_VALUE_ITERATION_HEADER_FILE

#include <tuple>
#include <iostream>
#include <iterator>

#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class applies the value iteration algorithm on a Model.
         */
        class ValueIteration {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * The discount parameter must be > 0.0 and <= 1.0,
                 * otherwise the constructor will throw an std::runtime_error.
                 *
                 * The epsilon parameter must be > 0.0,
                 * otherwise the constructor will throw an std::runtime_error.
                 *
                 * Note that the default value function size needs to match
                 * the number of states of the Model. Otherwise it will
                 * be ignored. An empty value function will be defaulted
                 * to all zeroes.
                 *
                 * @param discount The discount rate to use.
                 * @param epsilon The epsilon factor to stop the value iteration loop.
                 * @param maxIter The maximum number of iterations to perform.
                 * @param v The initial value function from which to start the loop.
                 */
                ValueIteration(double discount = 0.9, double epsilon = 0.01, unsigned maxIter = 0, ValueFunction v = ValueFunction(0) );

                /**
                 * @brief This function applies value iteration on an MDP to solve it.
                 *
                 * The algorithm is constrained by the currently set parameters.
                 *
                 * @tparam M The type of the solvable MDP.
                 * @param m The MDP that needs to be solved.
                 * @return A tuple containing a boolean value specifying the
                 *         return status of the solution problem, the
                 *         ValueFunction and the QFunction for the Model.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                std::tuple<bool, ValueFunction, QFunction> operator()(const M & m);

                /**
                 * @brief This function sets the discount parameter.
                 *
                 * The discount parameter must be > 0.0 and <= 1.0,
                 * otherwise the function will throw std::invalid_argument.
                 *
                 * @param d The new discount parameter.
                 */
                void setDiscount(double d);

                /**
                 * @brief This function sets the epsilon parameter.
                 *
                 * The epsilon parameter must be > 0.0,
                 * otherwise the function will do throw std::invalid_argument.
                 *
                 * @param e The new epsilon parameter.
                 */
                void setEpsilon(double e);

                /**
                 * @brief This function sets the max iteration parameter.
                 *
                 * @param m The new max iteration parameter.
                 */
                void setMaxIter(unsigned m);

                /**
                 * @brief This function sets the starting value function.
                 *
                 * An empty value function defaults to all zeroes. Note
                 * that the default value function size needs to match
                 * the number of states of the Model that needs to be
                 * solved. Otherwise it will be ignored.
                 *
                 * @param m The new starting value function.
                 */
                void setValueFunction(ValueFunction v);

                /**
                 * @brief This function will return the current set discount parameter.
                 *
                 * @return The currently set discount parameter.
                 */
                double getDiscount() const;

                /**
                 * @brief This function will return the currently set epsilon parameter.
                 *
                 * @return The currently set epsilon parameter.
                 */
                double getEpsilon() const;

                /**
                 * @brief This function will return the current set max iteration parameter.
                 *
                 * @return The currently set max iteration parameter.
                 */
                unsigned getMaxIter() const;

                /**
                 * @brief This function will return the current set default value function.
                 *
                 * @return The currently set default value function.
                 */
                const ValueFunction & getValueFunction() const;

            private:
                /**
                 * @brief This type represents the trivial part of a ValueFunction.
                 *
                 * This type contains, for each state-action pair, the expected
                 * one-step reward that can be gained. This does not include the
                 * non-trivial part, which is the inclusion of the future expected
                 * discounted value.
                 */
                using PRType = Table2D;

                // Parameters
                double discount_, epsilon_;
                unsigned maxIter_;
                ValueFunction vParameter_;

                // Internals
                ValueFunction v1_;
                size_t S, A;

                // Internal methods
                /**
                 * @brief This function computes the single PRType of the MDP once for improved speed.
                 *
                 * @tparam M The type of the solvable MDP.
                 * @param m The MDP that needs to be solved.
                 *
                 * @return The Models's PRType.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                PRType computePR(const M & model) const;

                /**
                 * @brief This function computes an upper bound on the number of iteration needed to solve the Model.
                 *
                 * @tparam M The type of the solvable MDP.
                 *
                 * @param m The MDP that needs to be solved.
                 * @param pr The Model's PRType.
                 *
                 * @return The estimated upper iteration bound.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                unsigned valueIterationBoundIter(const M & model, const PRType & pr) const;

                /**
                 * @brief This function creates the Model's most up-to-date QFunction.
                 *
                 * @tparam M The type of the solvable MDP.
                 *
                 * @param m The MDP that needs to be solved.
                 * @param pr The Model's PRType.
                 *
                 * @return A new QFunction.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                QFunction makeQFunction(const M & model, const PRType & pr) const;

                /**
                 * @brief This function applies a single pass Bellman operator, improving the current ValueFunction estimate.
                 *
                 * This function uses as base ValueFunction the one stored in
                 * the class (v1_). The result is then passed to vOut. This
                 * is to avoid allocating multiple ValueFunctions.
                 *
                 * @tparam M The type of the solvable MDP.
                 *
                 * @param m The MDP that needs to be solved.
                 * @param pr The Model's PRType.
                 * @param vOut The newly estimated ValueFunction.
                 */
                template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
                void bellmanOperator(const M & model, const PRType & pr, ValueFunction * vOut) const;
        };

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        std::tuple<bool, ValueFunction, QFunction> ValueIteration::operator()(const M & model) {
            // Extract necessary knowledge from model so we don't have to pass it around
            S = model.getS();
            A = model.getA();

            // Verify that parameter value function is compatible.
            if ( vParameter_.size() != S ) {
                if ( vParameter_.size() != 0 )
                    std::cerr << "AIToolbox: Size of starting value function in ValueIteration::solve() is incorrect, ignoring...\n";
                v1_ = ValueFunction(S, 0.0);
            }
            else
                v1_ = vParameter_;

            auto pr = computePR(model);
            {   // maxIter setup
                unsigned computedMaxIter = valueIterationBoundIter(model, pr);
                if ( !maxIter_ ) {
                    maxIter_ = discount_ != 1.0 ? computedMaxIter : 1000;
                }
                else {
                    maxIter_ = ( discount_ != 1.0 && maxIter_ > computedMaxIter ) ? computedMaxIter : maxIter_;
                }
            }
            // threshold setup
            double epsilon = ( discount_ != 1.0 ) ? ( epsilon_ * ( 1.0 - discount_ ) / discount_ ) : epsilon_;

            unsigned iter = 0;
            bool done = false, completed = false;

            ValueFunction v0 = v1_;

            while ( !done ) {
                ++iter;
                v0 = v1_;

                bellmanOperator( model, pr, &v1_ );

                std::transform(std::begin(v1_), std::end(v1_), std::begin(v0), std::begin(v0), std::minus<double>() );

                double variation;
                {
                    auto minmax = std::minmax_element(std::begin(v0), std::end(v0));
                    variation = *(minmax.second) - *(minmax.first);
                }
                if ( variation < epsilon ) {
                    completed = true;
                    done = true;
                }
                else if ( iter > maxIter_ ) {
                    done = true;
                }
            }
            // We do not guarantee that the Value/QFunctions are correct, as we stop as long as the policy is fine.
            return std::make_tuple(completed, v1_, makeQFunction(model, pr));
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        ValueIteration::PRType ValueIteration::computePR(const M & model) const {
            PRType pr(boost::extents[S][A]);

            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    for ( size_t s1 = 0; s1 < S; ++s1 )
                        pr[s][a] += model.getTransitionProbability(s,a,s1) * model.getExpectedReward(s,a,s1);

            return pr;
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        QFunction ValueIteration::makeQFunction(const M & model, const PRType & pr) const {
            QFunction q = pr;

            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    for ( size_t s1 = 0; s1 < S; ++s1 )
                        q[s][a] += model.getTransitionProbability(s,a,s1) * discount_ * v1_[s1];
            return q;
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        unsigned ValueIteration::valueIterationBoundIter(const M & model, const PRType & pr) const {
            std::vector<double> h(S, 0.0);

            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    for ( size_t s1 = 0; s1 < S; ++s1 )
                        h[s1] = std::min(h[s1], model.getTransitionProbability(s,a,s1));

            double k = 1.0 - std::accumulate(std::begin(h), std::end(h), 0.0);

            ValueFunction v(S);

            bellmanOperator(model, pr, &v);

            std::transform(std::begin(v), std::end(v), std::begin(v1_), std::begin(v), std::minus<double>() );

            double variation;
            {
                auto minmax = std::minmax_element(std::begin(v), std::end(v));
                variation = *(minmax.second) - *(minmax.first);
            }

            return std::ceil (
                    std::log( (epsilon_*(1.0-discount_)/discount_) / variation ) / std::log(discount_*k));
        }

        template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
        void ValueIteration::bellmanOperator(const M & model, const PRType & pr, ValueFunction * v) const {
            auto & vOut = *v;
            QFunction q = makeQFunction(model, pr);

            for ( size_t s = 0; s < S; ++s ) {
                // Accessing an element like this creates a temporary. Thus we need to bind it.
                decltype(q)::reference ref = q[s];
                auto it = std::max_element(std::begin(ref), std::end(ref));
                vOut[s] = *it;
            }
        }
    }
}

#endif
