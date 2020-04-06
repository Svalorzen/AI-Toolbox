#ifndef AI_TOOLBOX_FACTORED_BANDIT_UCVE_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_UCVE_HEADER_FILE

#include "AIToolbox/Factored/Bandit/Types.hpp"
#include <AIToolbox/Factored/Utils/GenericVariableElimination.hpp>

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
            struct Entry {
                V v;
                PartialAction tag;
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            };
            using Factor = std::vector<Entry>;

            using Result = std::tuple<Action, V>;
            using GVE = GenericVariableElimination<Factor>;

            /**
             * @brief This function is the entry point for the solving process.
             *
             * @tparam Iterable An iterable object containing a series of Entry.
             * @param inputRules The rules on which the Variable Elimination process will work.
             *
             * @return The optimal action to take, and its value.
             */
            template <typename Iterable>
            Result operator()(const Action & A, const double logtA, const Iterable & inputRules) {
                GVE::Graph graph(A.size());

                for (const Entry & rule : inputRules) {
                    const auto & [v, a] = rule;
                    auto & factorNode = graph.getFactor(a.first)->getData();
                    const auto id = toIndexPartial(A, a);

                    const auto it = std::lower_bound(
                        std::begin(factorNode),
                        std::end(factorNode),
                        id,
                        [](const auto & rule, size_t rhs) {return rule.first < rhs;}
                    );

                    if (it != std::end(factorNode) && it->first == id)
                        it->second[0].v += v;
                    else
                        factorNode.emplace(it, id, Factor{{v, PartialAction()}});
                }
                // Start solving process.
                return (*this)(A, logtA, graph);
            }

            /**
             * @brief This function performs the actual agent elimination process.
             *
             * @return The best action, randomly taken if multiple actions are eligible.
             */
            Result operator()(const Action & A, const double logtA, GVE::Graph & graph);
    };
}

#endif
