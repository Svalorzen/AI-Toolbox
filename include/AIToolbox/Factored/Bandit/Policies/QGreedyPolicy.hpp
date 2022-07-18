#ifndef AI_TOOLBOX_FACTORED_BANDIT_Q_GREEDY_POLICY_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_Q_GREEDY_POLICY_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Policies/PolicyInterface.hpp>
#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Utils/FilterMap.hpp>

#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/GraphUtils.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class implements a greedy policy through a QFunction.
     *
     * This class allows you to select effortlessly the best greedy actions
     * from a given list of QFunctionRules. In order to compute the best
     * action, or a given action probability the QGreedyPolicy must run
     * VariableElimination on the stored rules, so the process can get a
     * bit expensive.
     */
    template <typename Maximizer = VariableElimination>
    class QGreedyPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor with QFunctionRules.
             *
             * @param a The number of actions available to the agent.
             * @param q The QFunctionRules this policy is linked with.
             * @param ...args Parameters to pass to the maximizer on construction.
             */
            template <typename... Args>
            QGreedyPolicy(Action a, const FilterMap<QFunctionRule> & q, Args && ...args);

            /**
             * @brief Basic constructor with QFunction.
             *
             * @param a The number of actions available to the agent.
             * @param q The QFunction this policy is linked with.
             * @param ...args Parameters to pass to the maximizer on construction.
             */
            template <typename... Args>
            QGreedyPolicy(Action a, const QFunction & q, Args && ...args);

            /**
             * @brief This function chooses the greediest action for state s.
             *
             * @return The chosen action.
             */
            virtual Action sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * @param a The selected action.
             *
             * @return This function returns 1 if a is equal to the greediest action, and 0 otherwise.
             */
            virtual double getActionProbability(const Action & a) const override;

            /**
             * @brief This function returns a reference to the internal maximizer.
             *
             * This can be used to set the parameters of the chosen maximizer.
             */
            Maximizer & getMaximizer();

            /**
             * @brief This function returns a reference to the internal maximizer.
             */
            const Maximizer & getMaximizer() const;

            /**
             * @brief This function returns the currently set graph.
             */
            const typename Maximizer::Graph & getGraph() const;

        private:
            const FilterMap<QFunctionRule> * qc_;
            const QFunction * qm_;

            mutable Maximizer max_;
            mutable typename Maximizer::Graph graph_;
    };

    template <typename Maximizer>
    template <typename... Args>
    QGreedyPolicy<Maximizer>::QGreedyPolicy(Action a, const FilterMap<QFunctionRule> & q, Args && ...args) :
            Base(std::move(a)), qc_(&q), qm_(nullptr),
            max_(std::forward<Args>(args)...),
            graph_(MakeGraph<Maximizer>()(q, A))
    {}

    template <typename Maximizer>
    template <typename... Args>
    QGreedyPolicy<Maximizer>::QGreedyPolicy(Action a, const QFunction & q, Args && ...args) :
            Base(std::move(a)), qc_(nullptr), qm_(&q),
            max_(std::forward<Args>(args)...),
            graph_(MakeGraph<Maximizer>()(q, A))
    {}

    template <typename Maximizer>
    Action QGreedyPolicy<Maximizer>::sampleAction() const {
        if (qc_) {
            UpdateGraph<Maximizer>()(graph_, *qc_, A);
        } else {
            UpdateGraph<Maximizer>()(graph_, *qm_, A);
        }
        return std::get<0>(max_(A, graph_));
    }

    template <typename Maximizer>
    double QGreedyPolicy<Maximizer>::getActionProbability(const Action & a) const {
        if (veccmp(a, sampleAction()) == 0) return 1.0;
        return 0.0;
    }

    template <typename Maximizer>
    Maximizer & QGreedyPolicy<Maximizer>::getMaximizer() {
        return max_;
    }

    template <typename Maximizer>
    const Maximizer & QGreedyPolicy<Maximizer>::getMaximizer() const {
        return max_;
    }

    template <typename Maximizer>
    const typename Maximizer::Graph & QGreedyPolicy<Maximizer>::getGraph() const {
        return graph_;
    }
}

#endif
