#include <AIToolbox/FactoredMDP/Algorithms/Utils/MultiObjectiveVariableElimination.hpp>

#include <AIToolbox/Utils.hpp>

#include <iostream>

namespace AIToolbox {
    namespace FactoredMDP {
        using MOVE = MultiObjectiveVariableElimination;

        MOVE::Values crossSum(const MOVE::Values & lhs, const MOVE::Values & rhs);
        MOVE::Values crossSum(const MOVE::Values & lhs, const std::vector<const MOVE::Values*> rhs);
        std::vector<const MOVE::Values*> getPayoffs(const MOVE::Rules & rules, const PartialAction & jointAction);
        MOVE::Rules mergePayoffs(MOVE::Rules && lhs, MOVE::Rules && rhs);

        MOVE::MultiObjectiveVariableElimination(Action a) : graph_(a.size()), A(a) {}

        MOVE::Results MOVE::start() {
            // This can possibly be improved with some heuristic ordering
            while (graph_.agentSize())
                removeAgent(graph_.agentSize() - 1);

            Results retval;
            if (finalFactors_.size() == 0) return retval;

            for (const auto & fValue : finalFactors_)
                retval = crossSum(retval, fValue);

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
                        for (auto p : getPayoffs(factors[0]->f_.rules_, jointAction)) {
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
                            newValues = crossSum(newValues, getPayoffs(factors[i]->f_.rules_, jointAction));
                            std::cout << "- Crosssum, size is: " << newValues.size() << '\n';
                            // p3.prune(&newValues);
                        }

                        if (newValues.size() != 0) {
                            // Add tags
                            for (auto & nv : newValues) {
                                auto & first  = std::get<0>(nv).first;
                                auto & second = std::get<0>(nv).second;

                                size_t i = 0;
                                while (i < first.size() && first[i] < agent) ++i;

                                first.insert(std::begin(first) + i, agent);
                                second.insert(std::begin(second) + i, agentAction);
                            }
                            std::cout << "Added to values.\n";
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
                            std::get<0>(newRule) = removeFactor(jointAction, agent);

                            // Our insertion needs to be sorted so that we can
                            // merge efficiently later with rules in a factor.
                            // Doing insertion sort here is hopefully faster
                            // than quicksorting later the whole list.
                            auto comp = [](const Rule & lhs, const Rule & rhs) {
                                return veccmp(std::get<0>(lhs).second, std::get<0>(rhs).second) < 0;
                            };
                            auto pos = std::upper_bound(std::begin(newRules), std::end(newRules), newRule, comp);
                            newRules.emplace(pos, std::move(newRule));
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

                // Unfortunately here we cannot simply dump the new results in
                // the old factor as we do in the normal VariableElimination.
                // This is because in VariableElimination all elements are
                // summed together, which means that it doesn't matter whether
                // they are grouped or not. Here elements are CROSS-summed,
                // which means we cannot simply dump stuff lest losing a
                // cross-summing step.
                newFactor->f_.rules_ = mergePayoffs(std::move(newFactor->f_.rules_), std::move(newRules));
            }
        }

        std::vector<const MOVE::Values*> getPayoffs(const MOVE::Rules & rules, const PartialAction & jointAction) {
            std::vector<const MOVE::Values*> retval;
            for (const auto & rule : rules)
                if (match(jointAction, std::get<0>(rule)))
                    retval.push_back(&std::get<1>(rule));
            return retval;
        }

        MOVE::Rules mergePayoffs(MOVE::Rules && lhs, MOVE::Rules && rhs) {
            MOVE::Rules retval;
            // We're going to have at least these rules.
            retval.reserve(lhs.size() + rhs.size());

            // Here we merge two lists of Rules. What we want is that if any of
            // them match, we need to crossSum them. Otherwise, just bring them
            // over to the result list unchanged.
            size_t i = 0, j = 0;
            while (i < lhs.size() && j < rhs.size()) {
                auto first = veccmp(std::get<0>(lhs[i]).second, std::get<0>(rhs[j]).second);
                if (first < 0)
                    retval.emplace_back(std::move(lhs[i++]));
                else if (first > 0)
                    retval.emplace_back(std::move(rhs[j++]));
                else {
                    retval.emplace_back(std::get<0>(lhs[i]), crossSum(std::get<1>(lhs[i]), std::get<1>(rhs[j])));
                    ++i; ++j;
                }
            }
            // Copy remaining ones.
            for (; i < lhs.size(); ++i)
                retval.emplace_back(std::move(lhs[i]));
            for (; j < rhs.size(); ++j)
                retval.emplace_back(std::move(rhs[j]));

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
                    auto tags = merge(std::get<0>(lhsVal), std::get<0>(rhsVal));
                    auto values = std::get<1>(lhsVal) + std::get<1>(rhsVal);
                    retval.emplace_back(std::move(tags), std::move(values));
                }
            }
            return retval;
        }
    }
}
