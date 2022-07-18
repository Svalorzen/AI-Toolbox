#ifndef AI_TOOLBOX_FACTORED_MDP_Q_GREEDY_POLICY_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_Q_GREEDY_POLICY_HEADER_FILE

#include <AIToolbox/PolicyInterface.hpp>
#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/FilterMap.hpp>

#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>
#include <AIToolbox/Factored/MDP/Algorithms/Utils/GraphUtils.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class implements a greedy policy through a QFunction.
     *
     * This class allows you to select effortlessly the best greedy actions
     * from a given list of QFunctionRules, or from a QFunction.
     *
     * In order to compute the best action or a given action probability the
     * QGreedyPolicy must run VariableElimination on the stored rules, so the
     * process can get a bit expensive.
     */
    template <typename Maximizer = Bandit::VariableElimination>
    class QGreedyPolicy : public PolicyInterface<State, State, Action> {
        public:
            using Base = PolicyInterface<State, State, Action>;

            /**
             * @brief Basic constructor with QFunctionRules.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             * @param q The QFunctionRules this policy is linked with.
             * @param ...args Parameters to pass to the maximizer on construction.
             */
            template <typename... Args>
            QGreedyPolicy(State s, Action a, const FilterMap<QFunctionRule> & q, Args && ...args);

            /**
             * @brief Basic constructor with QFunction.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             * @param q The QFunction this policy is linked with.
             * @param ...args Parameters to pass to the maximizer on construction.
             */
            template <typename... Args>
            QGreedyPolicy(State s, Action a, const QFunction & q, Args && ...args);

            /**
             * @brief This function chooses the greediest action for state s.
             *
             * @param s The sampled state of the policy.
             *
             * @return The chosen action.
             */
            virtual Action sampleAction(const State & s) const override;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * @param s The selected state.
             * @param a The selected action.
             *
             * @return This function returns 1 if a is equal to the greediest action, and 0 otherwise.
             */
            virtual double getActionProbability(const State & s, const Action & a) const override;

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
    QGreedyPolicy<Maximizer>::QGreedyPolicy(State s, Action a, const FilterMap<QFunctionRule> & q, Args && ...args) :
            Base(std::move(s), std::move(a)), qc_(&q), qm_(nullptr),
            max_(std::forward<Args>(args)...),
            graph_(MakeGraph<Maximizer>()(q, A))
    {}

    template <typename Maximizer>
    template <typename... Args>
    QGreedyPolicy<Maximizer>::QGreedyPolicy(State s, Action a, const QFunction & q, Args && ...args) :
            Base(std::move(s), std::move(a)), qc_(nullptr), qm_(&q),
            max_(std::forward<Args>(args)...),
            graph_(MakeGraph<Maximizer>()(q, A))
    {}

    template <typename Maximizer>
    Action QGreedyPolicy<Maximizer>::sampleAction(const State & s) const {
        if (qc_) {
            UpdateGraph<Maximizer>()(graph_, qc_->filter(s), S, A, s);
        } else {
            UpdateGraph<Maximizer>()(graph_, *qm_, S, A, s);
        }
        return std::get<0>(max_(A, graph_));
    }

    template <typename Maximizer>
    double QGreedyPolicy<Maximizer>::getActionProbability(const State & s, const Action & a) const {
        if (veccmp(a, sampleAction(s)) == 0) return 1.0;
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
