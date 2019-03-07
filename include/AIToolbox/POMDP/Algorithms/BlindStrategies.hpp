#ifndef AI_TOOLBOX_POMDP_BLIND_STRATEGIES_HEADER_FILE
#define AI_TOOLBOX_POMDP_BLIND_STRATEGIES_HEADER_FILE

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Prune.hpp>

#include <AIToolbox/MDP/Utils.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the blind strategies lower bound.
     *
     * This class is useful in order to obtain a very simple lower bound for a
     * POMDP. The values for each action assume that the agent is always going
     * to take that same action forever afterwards.
     *
     * While this bound is somewhat loose, it can be a good starting point for
     * other algorithms as it's incredibly cheap to compute.
     *
     * We return the alphavectors for all actions. There's an incredibly high
     * likelihood that of the resulting alphavectors many are going to be
     * dominated, but we leave the pruning to the clients as maybe the
     * additional per-action information may be useful to somebody (and also
     * makes for easier testing ;) )
     */
    class BlindStrategies {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param horizon The maximum number of iterations to perform.
             * @param tolerance The tolerance factor to stop the value iteration loop.
             */
            BlindStrategies(unsigned horizon, double tolerance = 0.001);

            /**
             * @brief This function computes the blind strategies for the input POMDP.
             *
             * Here we return a simple VList for the specified horizon/tolerance.
             * Returning a ValueFunction would be pretty pointless, as the
             * implied policy here it's pretty obvious (always execute the same
             * action) so there's little sense in wrapping the bounds up.
             *
             * The bounds are still returned in a VList since at the moment
             * most POMDP utils expect this to work.
             *
             * An optional parameter for faster convengence can be specified.
             * If true, the algorithm won't initialize the values for each
             * action from zero, but from the minimum possible for that action
             * divided by 1 minus the model's discount (fixed so that division
             * by zero is impossible).
             *
             * This will make the algorithm converge faster, but the returned
             * values won't be the correct ones for the horizon specified (the
             * horizon will simply represent a bound on the number of iteration
             * performed by the algorithm.
             *
             * @param m The POMDP to be solved.
             * @param fasterConvergence Whether to initialize the internal
             *        vector for faster convergence.
             *
             * @return A tuple containing the maximum variation over all
             *         actions and the VList containing the found bounds.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            std::tuple<double, VList> operator()(const M & m, bool fasterConvergence);

            /**
             * @brief This function sets the tolerance parameter.
             *
             * The tolerance parameter must be >= 0.0, otherwise the function
             * will throw an std::invalid_argument. The tolerance parameter sets
             * the convergence criterion. A tolerance of 0.0 forces
             * BlindStrategies to perform a number of iterations equal to the
             * horizon specified. Otherwise, BlindStrategies will stop as soon
             * as the difference between two iterations is less than the
             * tolerance specified.
             *
             * @param tolerance The new tolerance parameter.
             */
            void setTolerance(double tolerance);

            /**
             * @brief This function sets the horizon parameter.
             *
             * @param h The new horizon parameter.
             */
            void setHorizon(unsigned h);

            /**
             * @brief This function returns the currently set toleranc parameter.
             *
             * @return The currently set tolerance parameter.
             */
            double getTolerance() const;

            /**
             * @brief This function returns the current horizon parameter.
             *
             * @return The currently set horizon parameter.
             */
            unsigned getHorizon() const;

        private:
            size_t horizon_;
            double tolerance_;
    };


    template <typename M, typename>
    std::tuple<double, VList> BlindStrategies::operator()(const M & m, const bool fasterConvergence) {
        const MDP::QFunction ir = [&]{
            if constexpr(MDP::is_model_eigen_v<M>) return m.getRewardFunction().transpose();
            else return MDP::computeImmediateRewards(m).transpose();
        }();
        // This function produces a very simple lower bound for the POMDP. The
        // bound for each action is computed assuming to take the same action forever
        // (so the bound for action 0 assumes to forever take action 0, the bound for
        // action 1 assumes to take action 1, etc.).
        VList retval;

        const bool useTolerance = checkDifferentSmall(tolerance_, 0.0);

        double maxVariation = 0.0;
        for (size_t a = 0; a < m.getA(); ++a) {
            auto newAlpha = Vector(m.getS());
            auto oldAlpha = Vector(m.getS());
            // Note that here we can take the minimum for each action
            // separately, since the implied policy will take that action
            // forever anyway so there cannot be "cross-pollination" between
            // different actions.
            if (fasterConvergence)
                oldAlpha.fill(ir.row(a).minCoeff() / std::max(0.0001, 1.0 - m.getDiscount()));
            else
                oldAlpha = ir.row(a);

            unsigned timestep = 0;
            double variation = tolerance_ * 2; // Make it bigger
            while ( timestep < horizon_ && ( !useTolerance || variation > tolerance_ ) ) {
                ++timestep;
                if constexpr(is_model_eigen_v<M>) {
                    newAlpha = ir.row(a) + (m.getDiscount() * m.getTransitionFunction(a) * oldAlpha).transpose();
                } else {
                    newAlpha = ir.row(a);
                    for (size_t s = 0; s < m.getS(); ++s) {
                        double sum = 0.0;
                        for (size_t s1 = 0; s1 < m.getS(); ++s1)
                            sum += m.getTransitionProbability(s, a, s1) * oldAlpha[s1];
                        newAlpha[s] += m.getDiscount() * sum;
                    }
                }

                if (useTolerance)
                    variation = (oldAlpha - newAlpha).cwiseAbs().maxCoeff();

                oldAlpha = std::move(newAlpha);
            }
            maxVariation = std::max(maxVariation, variation);
            retval.emplace_back(std::move(oldAlpha), a, VObs(0));
        }
        return std::make_tuple(useTolerance ? maxVariation : 0.0, std::move(retval));
    }
}

#endif
