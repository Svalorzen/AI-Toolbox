#ifndef AI_TOOLBOX_MDP_POLICY_ITERATION_HEADER_FILE
#define AI_TOOLBOX_MDP_POLICY_ITERATION_HEADER_FILE

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/Utils.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/MDP/Algorithms/Utils/PolicyEvaluation.hpp>

namespace AIToolbox {
    namespace MDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
        class PolicyIteration;
#endif

        template <typename M>
        class PolicyIteration<M> {
            public:
                PolicyIteration(unsigned horizon, double epsilon = 0.001);

                /**
                 * @brief This function applies policy iteration on an MDP to solve it.
                 *
                 * The algorithm is constrained by the currently set parameters.
                 *
                 * @tparam M The type of the solvable MDP.
                 * @param m The MDP that needs to be solved.
                 * @return A tuple containing a boolean value specifying whether
                 *         the specified epsilon bound was reached and the
                 *         ValueFunction and QFunction for the Model.
                 */
                QFunction operator()(const M & m);

                void setEpsilon(double e);
                void setHorizon(unsigned h);
                double getEpsilon() const;
                unsigned getHorizon() const;

            private:
                unsigned horizon_;
                double epsilon_;
        };

        template <typename M>
        PolicyIteration<M>::PolicyIteration(unsigned horizon, double epsilon) :
                horizon_(horizon)
        {
            setEpsilon(epsilon);
        }

        template <typename M>
        QFunction PolicyIteration<M>::operator()(const M & m) {
            const auto S = m.getS();
            const auto A = m.getA();

            PolicyEvaluation<M> eval(m, horizon_, epsilon_);

            auto qfun = makeQFunction(m.getS(), m.getA());
            QGreedyPolicy p(qfun);
            auto table = p.getPolicy();

            bool workToDo;
            do {
                workToDo = false;
                auto solution = eval(p);

                eval.setValues(std::move(std::get<1>(solution)));
                qfun = std::move(std::get<2>(solution));

                auto newTable = p.getPolicy();
                for (size_t s = 0; s < S; ++s) {
                    for (size_t a = 0; a < A; ++a) {
                        if (checkDifferentSmall(table(s,a), newTable(s,a))) {
                            workToDo = true;
                            table = std::move(newTable);
                            goto nextLoop;
                        }
                    }
                }
nextLoop:;
            } while (workToDo);

            return std::move(qfun);
        }

        template <typename M>
        void PolicyIteration<M>::setEpsilon(const double e) {
            if ( e < 0.0 ) throw std::invalid_argument("Epsilon must be >= 0");
            epsilon_ = e;
        }

        template <typename M>
        void PolicyIteration<M>::setHorizon(const unsigned h) {
            horizon_ = h;
        }

        template <typename M>
        double PolicyIteration<M>::getEpsilon()   const { return epsilon_; }

        template <typename M>
        unsigned PolicyIteration<M>::getHorizon() const { return horizon_; }
    }
}

#endif
