#ifndef AI_TOOLBOX_MDP_VALUE_ITERATION_HEADER_FILE
#define AI_TOOLBOX_MDP_VALUE_ITERATION_HEADER_FILE

#include <AIToolbox/MDP/Solver.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    class Policy;
    namespace MDP {
        /**
         * @brief This class applies the value iteration algorithm on a Model.
         */
        class ValueIteration : public Solver {
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
                 * @brief This function sets the discount parameter.
                 *
                 * The discount parameter must be > 0.0 and <= 1.0,
                 * otherwise the function will do nothing.
                 *
                 * @param d The new discount parameter.
                 */
                void setDiscount(double d);

                /**
                 * @brief This function sets the epsilon parameter.
                 *
                 * The epsilon parameter must be > 0.0,
                 * otherwise the function will do nothing.
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
                 * the number of states of the Model. Otherwise it will
                 * be ignored.
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
                 * @brief This function will return the current set epsilon parameter.
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

                /**
                 * @brief This function applies value iteration on the Model to solve it.
                 *
                 * The algorithm is constrained by the currently set parameters.
                 *
                 * @param m The Model that needs to be solved.
                 * @return A tuple containing a boolean value specifying the
                 *         return status of the solution problem, the
                 *         ValueFunction and the QFunction for the Model.
                 */
                virtual std::tuple<bool, ValueFunction, QFunction> operator()(const Model & m);
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
                const Model * model_;
                size_t S, A;
                // Internal methods
                /**
                 * @brief This function computes the single PRType of the Model once for improved speed.
                 * 
                 * @return The Models's PRType.
                 */
                PRType computePR() const;

                /**
                 * @brief This function computes an upper bound on the number of iteration needed to solve the Model.
                 *
                 * @param pr The Model's PRType.
                 *
                 * @return The estimated upper iteration bound.
                 */
                unsigned valueIterationBoundIter(const PRType & pr) const;

                /**
                 * @brief This function creates the Model's most up-to-date QFunction.
                 *
                 * @param pr The Model's PRType.
                 *
                 * @return A new QFunction.
                 */
                QFunction makeQFunction(const PRType & pr) const;

                /**
                 * @brief This function applies a single pass Bellman operator, improving the current ValueFunction estimate.
                 *
                 * This function uses as base ValueFunction the one stored in
                 * the class (v1_). The result is then passed to vOut. This
                 * is to avoid allocating multiple ValueFunctions.
                 *
                 * @param pr The Model's PRType.
                 * @param vOut The newly estimated ValueFunction.
                 */
                void bellmanOperator(const PRType & pr, ValueFunction & vOut) const;
        };

    }
}

#endif
