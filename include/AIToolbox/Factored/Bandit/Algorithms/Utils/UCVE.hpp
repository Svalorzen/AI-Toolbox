#ifndef AI_TOOLBOX_FACTORED_BANDIT_UCVE_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_UCVE_HEADER_FILE

#include "AIToolbox/Factored/Bandit/Types.hpp"
#include "AIToolbox/Factored/Utils/FactorGraph.hpp"

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class represents the UCVE process.
     *
     * This class implements variable elimination using bounds. It receives
     * as input a series of rules, one per PartialAction, containing both
     * an approximate value for the action, and a variance to keep track of
     * how much the rule could be actually worth.
     *
     * Internally, this uses a variation over
     * MultiObjectiveVariableElimination, where the two objectives are the
     * approximate value and the variance. Additionally, in order to be
     * more efficient, the agent removal process during Variable
     * Elimination computes an upper and lower variance bound for that
     * agent, in order to remove actions which cannot possibly be optimal
     * from the search.
     *
     * As this class doesn't really do much cleanup after solving a
     * process, it's probably best to create a separate instance for each
     * solving process.
     */
    class UCVE {
        public:
            // Estimated mean and inverse weighted counts
            using V = Eigen::Vector2d;
            // Tag - Vector pair
            using Entry = std::tuple<PartialAction, V>;
            using Entries = std::vector<Entry>;
            // Action -> (Tag + Vector)
            using Rule = std::tuple<PartialAction, Entries>;
            using Rules = std::vector<Rule>;

            using Result = Entry;

            struct Factor {
                Rules rules;
            };

            using Graph = FactorGraph<Factor>;

            /**
             * @brief Basic constructor.
             *
             * The constructor requires log(tA), which is the log of the
             * current timestep multiplied by the overall size of the
             * action space. This is required during the pruning process to
             * compute the exploration part due to the input variances for
             * each rule.
             *
             * @param A The action space of all agents.
             * @param logtA The current logtA.
             */
            UCVE(Action A, double logtA);

            /**
             * @brief This function is the entry point for the solving process.
             *
             * @tparam Iterable An iterable object containing a series of Entry.
             * @param inputRules The rules on which the Variable Elimination process will work.
             *
             * @return The optimal action to take, and its value.
             */
            template <typename Iterable>
            Result operator()(const Iterable & inputRules) {
                for (const Entry & rule : inputRules) {
                    const auto & a = std::get<0>(rule);
                    auto & rules = graph_.getFactor(a.first)->getData().rules;

                    rules.emplace_back(a, Entries{std::make_tuple(PartialAction(), std::get<1>(rule))};
                }
                // Start solving process.
                return start();
            }

        private:
            /**
             * @brief This function performs the actual agent elimination process.
             *
             * @return The best action, randomly taken if multiple actions are eligible.
             */
            Result start();

            /**
             * @brief This function performs a single step in the agent elimination process.
             *
             * First we compute the two bounds for the reward of the
             * agents. These bounds are then used in order to prune actions
             * which are neither promising nor good.
             *
             * @param agent The agent to remove from the graph.
             */
            void removeAgent(size_t agent);

            /**
             * @brief This function allows ordering and sorting of Rules to allow for merging.
             *
             * This function may only be used on Rules for the same agents.
             *
             * This function sorts rules by the actions taken by their
             * agents.
             *
             * @param lhs The left hand side.
             * @param rhs The right hand side.
             *
             * @return True if lhs comes before rhs, false otherwise.
             */
            static bool ruleComp(const Rule & lhs, const Rule & rhs);

            Action A;
            Graph graph_;
            std::vector<Entries> finalFactors_;
            double logtA_;
    };
}

#endif
