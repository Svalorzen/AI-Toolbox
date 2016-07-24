#ifndef AI_TOOLBOX_POMDP_QMDP_HEADER_FILE
#define AI_TOOLBOX_POMDP_QMDP_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Utils.hpp>

#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

namespace AIToolbox {
    namespace POMDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
        class QMDP;
#endif

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
         *
         * @tparam M The type of model that is solved by the algorithm.
         */
        template <typename M>
        class QMDP<M> {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * QMDP uses MDP::ValueIteration in order to solve the
                 * underlying MDP of the POMDP. Thus, its parameters are the
                 * same.
                 *
                 * @param horizon The maximum number of iterations to perform.
                 * @param epsilon The epsilon factor to stop the value iteration loop.
                 */
                QMDP(unsigned horizon, double epsilon = 0.001);

                /**
                 * @brief This function applies the QMDP algorithm on the input POMDP.
                 *
                 * This function computes the MDP::ValueFunction of the underlying
                 * MDP of the input POMDP with the parameters set. It then converts
                 * this solution into the equivalent POMDP::ValueFunction. Finally
                 * it returns both (plus the boolean specifying whether the epsilon
                 * constraint requested is satisfied or not).
                 *
                 * @tparam M The type of the input POMDP
                 * @param m The POMDP to be solved
                 *
                 * @return A tuple containing a boolean value specifying
                 *         whether the specified epsilon bound was reached, a
                 *         POMDP::ValueFunction and the equivalent MDP::ValueFunction.
                 */
                std::tuple<bool, ValueFunction, MDP::ValueFunction> operator()(const M & m);

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

            private:
                MDP::ValueIteration<M> solver_;
        };

        template <typename M>
        QMDP<M>::QMDP(unsigned horizon, double epsilon) : solver_(horizon, epsilon) {}

        template <typename M>
        std::tuple<bool, ValueFunction, MDP::ValueFunction> QMDP<M>::operator()(const M & m) {
            auto solution = solver_(m);
            auto & mdpValueFunction = std::get<1>(solution);
            auto & mdpValues  = std::get<MDP::VALUES >(mdpValueFunction);
            auto & mdpActions = std::get<MDP::ACTIONS>(mdpValueFunction);

            size_t S = m.getS();

            VList w;
            w.reserve(S);

            // We simply create a VList where each entry is a slice of the MDP::ValueFunction.
            // All in all, since an MDP is a subclass of POMDPs, what we are doing is writing the
            // true representation of the MDP solution (where we know only the solutions for the
            // corners of the belief space). The MDP::ValueFunction is simply a condensed form of
            // the solution, since in an MDP the only "beliefs" we have are the corners.
            for ( size_t s = 0; s < S; ++s ) {
                MDP::Values v(S); v.fill(0.0);
                v[s] = mdpValues[s];
                // All observations are 0 since we go back to the horizon 0 entry, which is nil.
                w.emplace_back(v, mdpActions[s], VObs(m.getO(), 0u));
            }

            ValueFunction vf(1, VList(1, makeVEntry(S)));
            vf.emplace_back(std::move(w));

            return std::make_tuple(std::get<0>(solution), vf, mdpValueFunction);
        }

        template <typename M>
        void QMDP<M>::setEpsilon(double e) {
            solver_.setEpsilon(e);
        }

        template <typename M>
        void QMDP<M>::setHorizon(unsigned h) {
            solver_.setHorizon(h);
        }

        template <typename M>
        double QMDP<M>::getEpsilon() const {
            return solver_.getEpsilon();
        }

        template <typename M>
        unsigned QMDP<M>::getHorizon() const {
            return solver_.getHorizon();
        }
    }
}

#endif
