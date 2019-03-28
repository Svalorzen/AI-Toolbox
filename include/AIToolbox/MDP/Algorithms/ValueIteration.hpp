#ifndef AI_TOOLBOX_MDP_VALUE_ITERATION_HEADER_FILE
#define AI_TOOLBOX_MDP_VALUE_ITERATION_HEADER_FILE

#include <AIToolbox/Impl/Logging.hpp>
#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>
#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::MDP {
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
     */
    class ValueIteration {
        public:
            /**
             * @brief Basic constructor.
             *
             * The tolerance parameter must be >= 0.0, otherwise the
             * constructor will throw an std::runtime_error. The tolerance
             * parameter sets the convergence criterion. A tolerance of 0.0
             * forces ValueIteration to perform a number of iterations
             * equal to the horizon specified. Otherwise, ValueIteration
             * will stop as soon as the difference between two iterations
             * is less than the tolerance specified.
             *
             * Note that the default value function size needs to match
             * the number of states of the Model. Otherwise it will
             * be ignored. An empty value function will be defaulted
             * to all zeroes.
             *
             * @param horizon The maximum number of iterations to perform.
             * @param tolerance The tolerance factor to stop the value iteration loop.
             * @param v The initial value function from which to start the loop.
             */
            ValueIteration(unsigned horizon, double tolerance = 0.001, ValueFunction v = {Values(), Actions(0)});

            /**
             * @brief This function applies value iteration on an MDP to solve it.
             *
             * The algorithm is constrained by the currently set parameters.
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

            /**
             * @brief This function sets the tolerance parameter.
             *
             * The tolerance parameter must be >= 0.0, otherwise the
             * constructor will throw an std::runtime_error. The tolerance
             * parameter sets the convergence criterion. A tolerance of 0.0
             * forces ValueIteration to perform a number of iterations
             * equal to the horizon specified. Otherwise, ValueIteration
             * will stop as soon as the difference between two iterations
             * is less than the tolerance specified.
             *
             * @param e The new tolerance parameter.
             */
            void setTolerance(double e);

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
             * @brief This function will return the currently set tolerance parameter.
             *
             * @return The currently set tolerance parameter.
             */
            double getTolerance() const;

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
            double tolerance_;
            unsigned horizon_;
            ValueFunction vParameter_;

            // Internals
            ValueFunction v1_;
    };

    template <typename M, typename>
    std::tuple<double, ValueFunction, QFunction> ValueIteration::operator()(const M & model) {
        // Extract necessary knowledge from model so we don't have to pass it around
        const size_t S = model.getS();
        const size_t A = model.getA();

        {
            // Verify that parameter value function is compatible.
            const size_t size = vParameter_.values.size();
            if ( size != S ) {
                if ( size != 0 ) {
                    AI_LOGGER(AI_SEVERITY_WARNING, "Size of starting value function is incorrect, ignoring...");
                }
                // Defaulting
                v1_ = makeValueFunction(S);
            }
            else
                v1_ = vParameter_;
        }

        const auto & ir = [&]{
            if constexpr (is_model_eigen_v<M>) return model.getRewardFunction();
            else return computeImmediateRewards(model);
        }();

        unsigned timestep = 0;
        double variation = tolerance_ * 2; // Make it bigger

        Values val0;
        auto & val1 = v1_.values;
        QFunction q = makeQFunction(S, A);

        const bool useTolerance = checkDifferentSmall(tolerance_, 0.0);
        while ( timestep < horizon_ && (!useTolerance || variation > tolerance_) ) {
            ++timestep;
            AI_LOGGER(AI_SEVERITY_DEBUG, "Processing timestep " << timestep);

            val0 = val1;

            // We apply the discount directly on the values vector.
            val1 *= model.getDiscount();
            q = computeQFunction(model, val1, ir);

            // Compute the new value function (note that also val1 is overwritten)
            bellmanOperatorInplace(q, &v1_);

            // We do this only if the tolerance specified is positive, otherwise we
            // continue for all the timesteps.
            if ( useTolerance )
                variation = (val1 - val0).cwiseAbs().maxCoeff();
        }

        // We do not guarantee that the Value/QFunctions are the perfect ones,
        // as we stop as within the given tolerance.
        return std::make_tuple(useTolerance ? variation : 0.0, std::move(v1_), std::move(q));
    }
}

#endif
