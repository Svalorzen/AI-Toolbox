#ifndef AI_TOOLBOX_MDP_VALUE_ITERATION_EIGEN_HEADER_FILE
#define AI_TOOLBOX_MDP_VALUE_ITERATION_EIGEN_HEADER_FILE

#include <tuple>
#include <iostream>
#include <iterator>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/Utils.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    namespace MDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<is_model_eigen<M>::value>::type>
        class ValueIterationEigen;
#endif

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
         * ValueFunction has converged within a certain accuracy, or the
         * horizon requested is reached.
         *
         * This implementation in particular is ported from the MATLAB
         * MDPToolbox (although it is simplified).
         *
         * This version of the algorithm is optimized to work with Eigen matrices.
         *
         * @tparam M The type of model that is solved by the algorithm.
         */
        template <typename M>
        class ValueIterationEigen<M> {
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
                ValueIterationEigen(unsigned horizon, double epsilon = 0.001, ValueFunction v = ValueFunction(Values(), Actions(0)));

                /**
                 * @brief This function applies value iteration on an MDP to solve it.
                 *
                 * The algorithm is constrained by the currently set parameters.
                 *
                 * @param m The MDP that needs to be solved.
                 * @return A tuple containing a boolean value specifying whether
                 *         the specified epsilon bound was reached and the
                 *         ValueFunction and the QFunction for the Model.
                 */
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

                /**
                 * @brief This function computes all immediate rewards (state and action) of the MDP once for improved speed.
                 *
                 * @param m The MDP that needs to be solved.
                 *
                 * @return The Models's immediate rewards.
                 */
                QFunction computeImmediateRewards(const M & model) const;

                /**
                 * @brief This function creates the Model's most up-to-date QFunction.
                 *
                 * @param m The MDP that needs to be solved.
                 * @param ir The immediate rewards of the model.
                 *
                 * @return A new QFunction.
                 */
                QFunction computeQFunction(const M & model, const QFunction & ir) const;

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

        template <typename M>
        ValueIterationEigen<M>::ValueIterationEigen(unsigned horizon, double epsilon, ValueFunction v) :
            horizon_(horizon), vParameter_(v),
            S(0), A(0)
        {
            setEpsilon(epsilon);
        }

        template <typename M>
        std::tuple<bool, ValueFunction, QFunction> ValueIterationEigen<M>::operator()(const M & model) {
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
                if ( useEpsilon )
                    variation = (val1 - val0).cwiseAbs().maxCoeff();
            }

            // We do not guarantee that the Value/QFunctions are the perfect ones, as we stop as within epsilon.
            return std::make_tuple(variation <= epsilon_, v1_, q);
        }

        template <typename M>
        QFunction ValueIterationEigen<M>::computeImmediateRewards(const M & model) const {
            QFunction pr = makeQFunction(S, A);

            for ( size_t a = 0; a < A; ++a )
                pr.col(a).noalias() = model.getTransitionFunction(a).cwiseProduct(model.getRewardFunction(a)) * Vector::Ones(S);

            return pr;
        }

        template <typename M>
        QFunction ValueIterationEigen<M>::computeQFunction(const M & model, const QFunction & ir) const {
            QFunction q(S, A);

            for ( size_t a = 0; a < A; ++a )
                q.col(a).noalias() = discount_ * ( model.getTransitionFunction(a).cwiseProduct(std::get<VALUES>(v1_).replicate(1, S).transpose() )) * Vector::Ones(S);

            q += ir;

            return q;
        }

        template <typename M>
        void ValueIterationEigen<M>::bellmanOperator(const QFunction & q, ValueFunction * v) const {
            assert(v);
            auto & values  = std::get<VALUES> (*v);
            auto & actions = std::get<ACTIONS>(*v);

            for ( size_t s = 0; s < S; ++s )
                values(s) = q.row(s).maxCoeff(&actions[s]);
        }

        template <typename M>
        void ValueIterationEigen<M>::setEpsilon(double e) {
            if ( e < 0.0 ) throw std::invalid_argument("Epsilon must be >= 0");
            epsilon_ = e;
        }

        template <typename M>
        void ValueIterationEigen<M>::setHorizon(unsigned h) {
            horizon_ = h;
        }

        template <typename M>
        void ValueIterationEigen<M>::setValueFunction(ValueFunction v) {
            vParameter_ = v;
        }

        template <typename M>
        double ValueIterationEigen<M>::getEpsilon()   const { return epsilon_; }

        template <typename M>
        unsigned ValueIterationEigen<M>::getHorizon() const { return horizon_; }

        template <typename M>
        const ValueFunction & ValueIterationEigen<M>::getValueFunction() const { return vParameter_; }
    }
}

#endif
