#ifndef AI_TOOLBOX_POMDP_QMDP_HEADER_FILE
#define AI_TOOLBOX_POMDP_QMDP_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Utils.hpp>

#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the QMDP algorithm.
     *
     * QMDP is a particular way to approach a POMDP problem and solve it
     * approximately. The idea is to compute a solution that disregards the
     * partial observability for all timesteps but the next one. Thus, we
     * assume that after the next action the agent will suddenly be able to
     * see the true state of the environment, and act accordingly. In doing
     * so then, it will use an MDP value function.
     *
     * Remember that only the solution process acts this way. When time to
     * act the QMDP solution is simply applied at every timestep, every
     * time assuming that the partial observability is going to last one
     * step.
     *
     * All in all, this class is pretty much a converter of an
     * MDP::ValueFunction into a POMDP::ValueFunction.
     *
     * Although the solution is approximate and overconfident (since we
     * assume that partial observability is going to go away, we think we
     * are going to get more reward), it is still good to obtain a closer
     * upper bound on the true solution. This can be used, for example, to
     * boost bounds on online methods, decreasing the time they take to
     * converge.
     *
     * The solution returned by QMDP will thus have only horizon 1, since
     * the horizon requested is implicitly encoded in the MDP part of the
     * solution.
     */
    class QMDP {
        public:
            /**
             * @brief Basic constructor.
             *
             * QMDP uses MDP::ValueIteration in order to solve the
             * underlying MDP of the POMDP. Thus, its parameters (and
             * bounds) are the same.
             *
             * @param horizon The maximum number of iterations to perform.
             * @param tolerance The tolerance factor to stop the value iteration loop.
             */
            QMDP(unsigned horizon, double tolerance = 0.001);

            /**
             * @brief This function applies the QMDP algorithm on the input POMDP.
             *
             * This function computes the MDP::QFunction of the underlying MDP
             * of the input POMDP with the parameters set using ValueIteration.
             *
             * It then converts this solution into the equivalent
             * POMDP::ValueFunction. Finally it returns both (plus the
             * variation for the last iteration of ValueIteration).
             *
             * Note that no pruning is performed here, so some vectors might be
             * dominated.
             *
             * @param m The POMDP to be solved
             *
             * @return A tuple containing the maximum variation for the
             *         ValueFunction, the computed ValueFunction and the
             *         equivalent MDP::QFunction.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            std::tuple<double, ValueFunction, MDP::QFunction> operator()(const M & m);

            /**
             * @brief This function converts an MDP::QFunction into the equivalent POMDP VList.
             *
             * This function directly converts a QFunction into the equivalent
             * VList.
             *
             * The function needs to know the observation space so that if
             * needed the output can be used in a ValueFunction, and possibly
             * with a Policy, without crashing.
             *
             * @param O The observation space of the POMDP to make a VList for.
             * @param qfun The MDP QFunction from which to create a VList.
             *
             * @return A VList equivalent to the input QFunction.
             */
            static VList fromQFunction(size_t O, const MDP::QFunction & qfun);

            /**
             * @brief This function sets the tolerance parameter.
             *
             * The tolerance parameter must be >= 0.0, otherwise the function
             * will throw an std::invalid_argument. The tolerance parameter
             * sets the convergence criterion. A tolerance of 0.0 forces the
             * internal ValueIteration to perform a number of iterations
             * equal to the horizon specified. Otherwise, ValueIteration
             * will stop as soon as the difference between two iterations
             * is less than the tolerance specified.
             *
             * @param t The new tolerance parameter.
             */
            void setTolerance(double t);

            /**
             * @brief This function sets the horizon parameter.
             *
             * @param h The new horizon parameter.
             */
            void setHorizon(unsigned h);

            /**
             * @brief This function returns the currently set tolerance parameter.
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
            MDP::ValueIteration solver_;
    };

    template <typename M, typename>
    std::tuple<double, ValueFunction, MDP::QFunction> QMDP::operator()(const M & m) {
        auto solution = solver_(m);

        const size_t S = m.getS();

        auto v = makeValueFunction(S);
        v.emplace_back(fromQFunction(m.getO(), std::get<2>(solution)));

        return std::make_tuple(std::get<0>(solution), v, std::move(std::get<2>(solution)));
    }
}

#endif
