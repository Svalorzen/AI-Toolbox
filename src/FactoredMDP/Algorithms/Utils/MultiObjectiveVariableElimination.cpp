#include <AIToolbox/FactoredMDP/Algorithms/Utils/MultiObjectiveVariableElimination.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        using MOVE = MultiObjectiveVariableElimination;

        MOVE::Values crossSum(const MOVE::Values & lhs, const MOVE::Values & rhs);
        MOVE::Values crossSum(const MOVE::Values & lhs, const std::vector<const MOVE::Values*> rhs);
        std::vector<const MOVE::Values*> getPayoffs(const MOVE::Factor & factor, const PartialAction & jointAction);

        MOVE::MultiObjectiveVariableElimination(Action a) : graph_(a.size()), A(a) {}

        MOVE::Results MOVE::start() {
            // This can possibly be improved with some heuristic ordering
            while (graph_.agentSize())
                removeAgent(graph_.agentSize() - 1);

            Results retval;
            if (finalFactors_.size() == 0) return retval;

            Values newValues = finalFactors_[0];
            for (const auto & fValue : finalFactors_)
                newValues = crossSum(newValues, fValue);

            // convert(newValues -> retval);
            // p1.prune(&retval);
            return retval;
        }

        void MOVE::removeAgent(size_t agent) {
            auto factors = graph_.getNeighbors(agent);
            auto agents = graph_.getNeighbors(factors);

            Rules newRules;
            PartialFactorsEnumerator jointActions(A, agents, agent);
            auto id = jointActions.getFactorToSkipId();

            if (factors.size() > 0) {
                while (jointActions.isValid()) {
                    auto & jointAction = *jointActions;

                    Rule newRule;
                    auto & values = std::get<1>(newRule);
                    for (size_t agentAction = 0; agentAction < A[agent]; ++agentAction) {
                        jointAction.second[id] = agentAction;

                        Values newValues;
                        for (auto p : getPayoffs(factors[0]->f_, jointAction))
                            newValues.insert(std::end(newValues), std::begin(*p), std::end(*p));
                        // So the idea here is that we are computing results for
                        // this particular subset of agents. Here we are working
                        // with a single action. However, we may have eliminated
                        // agents already. This means that this factor will contain
                        // a certain number of rules, which depend on different
                        // "already taken" actions of the eliminated agents.
                        //
                        // During normal VE, we can simply add up all tags since
                        // they can't possibly conflict (due to the max operator
                        // which always makes us pick the best one). Here instead,
                        // payoffs returned by the getPayoffs function can't get
                        // squashed into a single one and summed, since their tags
                        // are no longer guaranteed unique.
                        //
                        // Thus we get them all, and during the cross/sum we create
                        // even more rules, joining their tags together, and
                        // possibly merge them if we see equal ones.
                        for (size_t i = 1; i < factors.size(); ++i) {
                            newValues = crossSum(newValues, getPayoffs(factors[i]->f_, jointAction));
                            // p3.prune(&newValues);
                        }
                        if (newValues.size() != 0) {
                            values.insert(std::end(values), std::make_move_iterator(std::begin(newValues)),
                                                            std::make_move_iterator(std::end(newValues)));
                        }
                    }
                    // p2.prune(&values);
                    if (values.size() != 0) {
                        // If this is a final factor we do the alternative path
                        // here, to avoid copying joint actions which we won't
                        // really need anymore.
                        if (agents.size() > 1) {
                            std::get<0>(newRule) = jointAction;
                            newRules.emplace_back(std::move(newRule));
                        } else {
                            finalFactors_.emplace_back(std::move(values));
                        }
                    }
                    jointActions.advance();
                }
            }

            for (auto & it : factors)
                graph_.erase(it);
            graph_.erase(agent);

            if (newRules.size() == 0) return;
            if (agents.size() > 1) {
                agents.erase(std::remove(std::begin(agents), std::end(agents), agent), std::end(agents));

                auto newFactor = graph_.getFactor(agents);
                newFactor->f_.rules_.insert(
                        std::end(newFactor->f_.rules_),
                        std::make_move_iterator(std::begin(newRules)),
                        std::make_move_iterator(std::end(newRules))
                );
            }
        }

        std::vector<const MOVE::Values*> getPayoffs(const MOVE::Factor & factor, const PartialAction & jointAction) {
            std::vector<const MOVE::Values*> retval;
            for (const auto & rule : factor.rules_)
                if (match(jointAction, std::get<0>(rule)))
                    retval.push_back(&std::get<1>(rule));
            return retval;
        }

        MOVE::Values crossSum(const MOVE::Values & lhs, const std::vector<const MOVE::Values*> rhs) {
            MOVE::Values retval;
            for (auto p : rhs) {
                auto tmp = crossSum(lhs, *p);
                retval.insert(std::end(retval), std::make_move_iterator(std::begin(tmp)),
                                                std::make_move_iterator(std::end(tmp)));
            }
            return retval;
        }

        MOVE::Values crossSum(const MOVE::Values & lhs, const MOVE::Values & rhs) {
            MOVE::Values retval;
            retval.reserve(lhs.size() + rhs.size());
            // We do the rhs first since they'll usually be shorter (due to
            // this class usage), so hopefully we can use the cache better.
            for (const auto & rhsVal : rhs) {
                for (const auto & lhsVal : lhs) {
                    auto values = lhsVal.first + rhsVal.first;
                    auto tags = merge(lhsVal.second, rhsVal.second);
                    retval.emplace_back(std::move(values), std::move(tags));
                }
            }
            return retval;
        }
    }
}
