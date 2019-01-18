#ifndef AI_TOOLBOX_MDP_POLICY_EVALUATION_HEADER_FILE
#define AI_TOOLBOX_MDP_POLICY_EVALUATION_HEADER_FILE

#include <tuple>
#include <iterator>

#include <AIToolbox/Impl/Logging.hpp>
#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/MDP/Policies/PolicyInterface.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class applies the policy evaluation algorithm on a policy.
     *
     * Policy Evaluation computes the values and QFunction for a particular
     * policy used on a given Model.
     *
     * This class is setup so it is easy to reuse on multiple policies
     * using the same Model, so that no redundant computations have to be
     * performed.
     *
     * @tparam M The type of model that is solved by the algorithm.
     */
    template <typename M>
    class PolicyEvaluation {
        static_assert(is_model_v<M>, "This class only works for MDP models!");

        public:
            /**
             * @brief Basic constructor.
             *
             * The tolerance parameter must be >= 0.0, otherwise the
             * constructor will throw an std::runtime_error. The tolerance
             * parameter sets the convergence criterion. A tolerance of 0.0
             * forces PolicyEvaluation to perform a number of iterations
             * equal to the horizon specified. Otherwise, PolicyEvaluation
             * will stop as soon as the difference between two iterations
             * is less than the tolerance specified.
             *
             * Note that the default value function size needs to match
             * the number of states of the Model. Otherwise it will
             * be ignored. An empty value function will be defaulted
             * to all zeroes.
             *
             * @param m The MDP to evaluate a policy for.
             * @param horizon The maximum number of iterations to perform.
             * @param tolerance The tolerance factor to stop the policy evaluation loop.
             * @param v The initial value function from which to start the loop.
             */
            PolicyEvaluation(const M & m, unsigned horizon, double tolerance = 0.001, Values v = Values());

            /**
             * @brief This function applies policy evaluation on a policy.
             *
             * The algorithm is constrained by the currently set parameters.
             *
             * @param p The policy to be evaluated.
             * @return A tuple containing the maximum variation for the
             *         ValueFunction, the ValueFunction and the QFunction for
             *         the Model and policy.
             */
            std::tuple<double, Values, QFunction> operator()(const PolicyInterface & p);

            /**
             * @brief This function sets the tolerance parameter.
             *
             * The tolerance parameter must be >= 0.0, otherwise the
             * constructor will throw an std::runtime_error. The tolerance
             * parameter sets the convergence criterion. A tolerance of 0.0
             * forces PolicyEvaluation to perform a number of iterations
             * equal to the horizon specified. Otherwise, PolicyEvaluation
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
            void setValues(Values v);

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
             * @brief This function will return the currently set default values.
             *
             * @return The currently set default values.
             */
            const Values & getValues() const;

        private:
            // Parameters
            double tolerance_;
            unsigned horizon_;
            Values vParameter_;
            const M & model_;

            // Internals
            QFunction immediateRewards_;
            Values v1_;
            size_t S, A;
    };

    template <typename M>
    PolicyEvaluation<M>::PolicyEvaluation(const M & m, const unsigned horizon, const double tolerance, Values v) :
            horizon_(horizon), vParameter_(std::move(v)), model_(m), S(0), A(0)
    {
        setTolerance(tolerance);

        // Extract necessary knowledge from model so we don't have to pass it around
        S = model_.getS();
        A = model_.getA();

        // Only compute the immediate rewards if we need them.
        if constexpr (!is_model_eigen_v<M>)
            immediateRewards_ = computeImmediateRewards(m);
    }

    template <typename M>
    std::tuple<double, Values, QFunction> PolicyEvaluation<M>::operator()(const PolicyInterface & policy) {
        {
            // Verify that parameter value function is compatible.
            const size_t size = vParameter_.size();
            if ( size != S ) {
                if ( size != 0 ) {
                    AI_LOGGER(AI_SEVERITY_WARNING, "Size of starting value function is incorrect, ignoring...");
                }
                // Defaulting
                v1_ = Values(S);
                v1_.setZero();
            }
            else
                v1_ = vParameter_;
        }

        unsigned timestep = 0;
        double variation = tolerance_ * 2; // Make it bigger

        Values val0;
        QFunction q = makeQFunction(S, A);
        const auto p = policy.getPolicy();

        const bool useTolerance = checkDifferentSmall(tolerance_, 0.0);
        while ( timestep < horizon_ && (!useTolerance || variation > tolerance_) ) {
            ++timestep;
            AI_LOGGER(AI_SEVERITY_DEBUG, "Processing timestep " << timestep);

            val0 = v1_;

            // We apply the discount directly on the values vector.
            v1_ *= model_.getDiscount();
            // We use the implicit reward function if it is available,
            // otherwise we use the one we computed beforehand.
            if constexpr(is_model_eigen_v<M>)
                q = computeQFunction(model_, v1_, model_.getRewardFunction());
            else
                q = computeQFunction(model_, v1_, immediateRewards_);

            // Compute the values for this policy
            for ( size_t s = 0; s < S; ++s )
                v1_(s) = q.row(s) * p.row(s).transpose();

            // We do this only if the tolerance specified is positive,
            // otherwise we continue for all the timesteps.
            if ( useTolerance )
                variation = (v1_ - val0).cwiseAbs().maxCoeff();
        }

        // We do not guarantee that the Value/QFunctions are the perfect
        // ones, as we stop within the input tolerance.
        return std::make_tuple(useTolerance ? variation : 0.0, std::move(v1_), std::move(q));
    }

    template <typename M>
    void PolicyEvaluation<M>::setTolerance(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Tolerance must be >= 0");
        tolerance_ = t;
    }

    template <typename M>
    void PolicyEvaluation<M>::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    template <typename M>
    void PolicyEvaluation<M>::setValues(Values v) {
        vParameter_ = std::move(v);
    }

    template <typename M>
    double PolicyEvaluation<M>::getTolerance()   const { return tolerance_; }

    template <typename M>
    unsigned PolicyEvaluation<M>::getHorizon() const { return horizon_; }

    template <typename M>
    const Values & PolicyEvaluation<M>::getValues() const { return vParameter_; }
}

#endif
