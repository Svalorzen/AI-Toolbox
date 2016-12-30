#include <AIToolbox/FactoredMDP/Algorithms/Utils/MultiObjectiveVariableElimination.hpp>

#include <AIToolbox/Utils.hpp>

#include <iostream>

namespace AIToolbox {
    namespace FactoredMDP {
        using MOVE = MultiObjectiveVariableElimination;

        MOVE::Values crossSum(const MOVE::Values & lhs, const MOVE::Values & rhs);
        MOVE::Values crossSum(const MOVE::Values & lhs, const std::vector<const MOVE::Values*> rhs);
        std::vector<const MOVE::Values*> getPayoffs(const MOVE::Factor & factor, const PartialAction & jointAction);
        MOVE::Rules mergePayoffs(const MOVE::Factor & factor, MOVE::Rules && rules);

        MOVE::MultiObjectiveVariableElimination(Action a) : graph_(a.size()), A(a) {}

        MOVE::Results MOVE::start() {
            // This can possibly be improved with some heuristic ordering
            while (graph_.agentSize())
                removeAgent(graph_.agentSize() - 1);

            Results retval;
            if (finalFactors_.size() == 0) return retval;
            std::cout << "We have " << finalFactors_.size() << " FFactors\n";

            Values newValues;
            for (const auto & fValue : finalFactors_) {
                std::cout << "This ff has " << fValue.size() << " entries.\n";
                newValues = crossSum(newValues, fValue);
                std::cout << "After crossumming, " << newValues.size() << " entries\n";
            }

            for (const auto & v : newValues)
                retval.emplace_back(toFactors(A.size(), v.second), std::move(v.first));
            // p1.prune(&retval);
            return retval;
        }

        void MOVE::removeAgent(size_t agent) {
            auto factors = graph_.getNeighbors(agent);
            auto agents = graph_.getNeighbors(factors);
            std::cout << "### REMOVING AGENT " << agent << '\n';
            std::cout << "### CONSIDERING " << factors.size() << " FACTORS\n";

            Rules newRules;
            PartialFactorsEnumerator jointActions(A, agents, agent);
            auto id = jointActions.getFactorToSkipId();

            if (factors.size() > 0) {
                while (jointActions.isValid()) {
                    auto & jointAction = *jointActions;

                    Rule newRule;
                    auto & values = std::get<1>(newRule);
                    for (size_t agentAction = 0; agentAction < A[agent]; ++agentAction) {
                        std::cout << "/////// New joint action\n";
                        jointAction.second[id] = agentAction;

                        Values newValues;
                        for (auto p : getPayoffs(factors[0]->f_, jointAction)) {
                            std::cout << "Inserting start?\n";
                            newValues.insert(std::end(newValues), std::begin(*p), std::end(*p));
                        }
                        std::cout << "New value, size: " << newValues.size() << "\n";
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
                            std::cout << "- Crosssum, size is: " << newValues.size() << '\n';
                            // p3.prune(&newValues);
                        }
                        if (newValues.size() != 0) {
                            // Add tags
                            std::cout << "Adding tags...";
                            for (auto & nv : newValues) {
                                auto & first  = nv.second.first;
                                auto & second = nv.second.second;

                                size_t i = 0;
                                while (i < first.size() && first[i] < agent) ++i;

                                first.insert(std::begin(first) + i, agent);
                                second.insert(std::begin(second) + i, agentAction);
                                std::cout << "tag added - ";
                            }
                            std::cout << "Added to values.\n";
                            values.insert(std::end(values), std::make_move_iterator(std::begin(newValues)),
                                                            std::make_move_iterator(std::end(newValues)));
                        }
                    }
                    // p2.prune(&values);
                    if (values.size() != 0) {
                        std::cout << "Pushing: ";
                        // If this is a final factor we do the alternative path
                        // here, to avoid copying joint actions which we won't
                        // really need anymore.
                        if (agents.size() > 1) {
                            std::cout << "To new rules.\n";
                            std::get<0>(newRule) = removeFactor(jointAction, agent);

                            std::cout << "Joint Action: [";
                            for (const auto x : std::get<0>(newRule).second)
                                std::cout << x << ", ";
                            std::cout << "] -->\n";
                            for (const auto & aa : std::get<1>(newRule)) {
                                std::cout << "    [";
                                for (const auto & ll : aa.second.second)
                                    std::cout << ll << ", ";
                                std::cout << "] ==> [";
                                std::cout << std::get<0>(aa).transpose() << "]\n";
                            }

                            newRules.emplace_back(std::move(newRule));
                        } else {
                            std::cout << "To FF\n";
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
                std::cout << "Moving to new factor\n";
                agents.erase(std::remove(std::begin(agents), std::end(agents), agent), std::end(agents));

                auto newFactor = graph_.getFactor(agents);
                // Unfortunately here we cannot simply dump the new results in
                // the old factor as we do in the normal VariableElimination.
                // This is because in VariableElimination all elements are
                // summed together, which means that it doesn't matter whether
                // they are grouped or not. Here elements are CROSS-summed,
                // which means we cannot simply dump stuff lest losing a
                // cross-summing step.
                newFactor->f_.rules_ = mergePayoffs(newFactor->f_, std::move(newRules));
            }
        }

        std::vector<const MOVE::Values*> getPayoffs(const MOVE::Factor & factor, const PartialAction & jointAction) {
            std::vector<const MOVE::Values*> retval;
            for (const auto & rule : factor.rules_)
                if (match(jointAction, std::get<0>(rule)))
                    retval.push_back(&std::get<1>(rule));
            std::cout << "GetPayoff returns " << retval.size() << " rules.\n";
            return retval;
        }

        MOVE::Rules mergePayoffs(const MOVE::Factor & factor, MOVE::Rules && rules) {
            MOVE::Rules retval;
            retval.reserve(rules.size());
            for (auto & rr : rules) {
                bool found = false;
                for (const auto & rule : factor.rules_) {
                    if (veccmp(std::get<0>(rr).second, std::get<0>(rule).second) == 0) {
                        found = true;
                        retval.emplace_back(std::get<0>(rr), crossSum(std::get<1>(rr), std::get<1>(rule)));
                        break;
                    }
                }
                if (!found)
                    retval.emplace_back(std::move(rr));
            }

            std::cout << "GetPayoff returns " << retval.size() << " rules.\n";
            return retval;
        }

        MOVE::Values crossSum(const MOVE::Values & lhs, const std::vector<const MOVE::Values*> rhs) {
            if (!rhs.size()) return lhs;

            MOVE::Values retval;
            for (auto p : rhs) {
                auto tmp = crossSum(lhs, *p);
                retval.insert(std::end(retval), std::make_move_iterator(std::begin(tmp)),
                                                std::make_move_iterator(std::end(tmp)));
            }
            return retval;
        }

        MOVE::Values crossSum(const MOVE::Values & lhs, const MOVE::Values & rhs) {
            if (!lhs.size()) return rhs;
            if (!rhs.size()) return lhs;
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
