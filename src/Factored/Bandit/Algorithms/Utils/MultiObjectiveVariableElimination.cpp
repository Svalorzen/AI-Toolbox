#include <AIToolbox/Factored/Bandit/Algorithms/Utils/MultiObjectiveVariableElimination.hpp>

#include <algorithm>

#include <boost/iterator/transform_iterator.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Prune.hpp>

namespace AIToolbox::Factored::Bandit {
    using MOVE = MultiObjectiveVariableElimination;

    /**
     * @brief This function cross-sums the input lists.
     *
     * For each element of tag/value in both inputs, a new value will be
     * returned with a value equal to the element-wise sum of the operands,
     * and tag equal to merged tags of the operands.
     *
     * @param lhs The left hand side.
     * @param rhs The right hand side.
     *
     * @return A new list containing all cross-sums.
     */
    MOVE::Entries crossSum(const MOVE::Entries & lhs, const MOVE::Entries & rhs);

    /**
     * @brief This function cross-sums the input lists.
     *
     * Cross-sums are performed considering the right-hand side as one
     * single joined list. This is useful considering how the getPayoffs()
     * function works.
     *
     * \sa crossSum(const MOVE::Entries &, const MOVE::Entries &);
     *
     * @param lhs The left hand side.
     * @param rhs A list of pointers to valid Entries lists.
     *
     * @return A new list containing all cross-sums.
     */
    MOVE::Entries crossSum(const MOVE::Entries & lhs, const std::vector<const MOVE::Entries*> rhs);

    /**
     * @brief This function returns a list of pointers to all Entries from the Rules matching the input joint action.
     *
     * @param rules A list of Rule.
     * @param jointAction A joint action to match Rules against.
     *
     * @return A list of pointers to the Entries contained in the Rules matched against the input action.
     */
    std::vector<const MOVE::Entries*> getPayoffs(const MOVE::Rules & rules, const PartialAction & jointAction);

    /**
     * @brief This function returns cross-sums common elements between the input plus all unique Rules.
     *
     * The inputs must be sorted by PartialAction lexically. This function
     * moves from its inputs.
     *
     * @param lhs The left hand side.
     * @param rhs The right hand side.
     *
     * @return A list of cross-summed rules.
     */
    MOVE::Rules mergePayoffs(MOVE::Rules && lhs, MOVE::Rules && rhs);

    // -----------------------

    MOVE::MultiObjectiveVariableElimination(Action a) : A(std::move(a)), graph_(A.size()) {}

    MOVE::Results MOVE::start() {
        // This can possibly be improved with some heuristic ordering
        while (graph_.variableSize())
            removeAgent(graph_.variableSize() - 1);

        Results retval;
        if (finalFactors_.size() == 0) return retval;

        for (const auto & fValue : finalFactors_)
            retval = crossSum(retval, fValue);

        // P1 pruning
        const auto unwrap = +[](Entry & e) -> Rewards & {return std::get<1>(e);};
        const auto rbegin = boost::make_transform_iterator(std::begin(retval), unwrap);
        const auto rend   = boost::make_transform_iterator(std::end  (retval), unwrap);

        retval.erase(AIToolbox::extractDominated(unwrap(retval[0]).size(), rbegin, rend).base(), std::end(retval));

        return retval;
    }

    void MOVE::removeAgent(const size_t agent) {
        const auto factors = graph_.getNeighbors(agent);
        auto agents = graph_.getNeighbors(factors);

        Rules newRules;
        PartialFactorsEnumerator jointActions(A, agents, agent);
        const auto id = jointActions.getFactorToSkipId();

        const bool isFinalFactor = agents.size() == 1;

        while (jointActions.isValid()) {
            auto & jointAction = *jointActions;

            Entries values;
            for (size_t agentAction = 0; agentAction < A[agent]; ++agentAction) {
                jointAction.second[id] = agentAction;

                Entries newEntries;
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
                for (size_t i = 0; i < factors.size(); ++i) {
                    newEntries = crossSum(newEntries, getPayoffs(factors[i]->getData().rules_, jointAction));
                    // p3.prune(&newEntries);
                }

                if (newEntries.size() != 0) {
                    // Add tags
                    for (auto & nv : newEntries) {
                        auto & first  = std::get<0>(nv).first;
                        auto & second = std::get<0>(nv).second;

                        size_t i = 0;
                        while (i < first.size() && first[i] < agent) ++i;

                        first.insert(std::begin(first) + i, agent);
                        second.insert(std::begin(second) + i, agentAction);
                    }
                    values.insert(std::end(values), std::make_move_iterator(std::begin(newEntries)),
                                                    std::make_move_iterator(std::end(newEntries)));
                }
            }

            // p2.prune(&values);

            if (values.size() != 0) {
                // If this is a final factor we do the alternative path
                // here, to avoid copying joint actions which we won't
                // really need anymore.
                if (!isFinalFactor) {
                    newRules.emplace_back(removeFactor(jointAction, agent), std::move(values));
                } else {
                    finalFactors_.emplace_back(std::move(values));
                }
            }
            jointActions.advance();
        }

        for (auto & it : factors)
            graph_.erase(it);
        graph_.erase(agent);

        if (newRules.size() == 0) return;
        if (!isFinalFactor) {
            agents.erase(std::remove(std::begin(agents), std::end(agents), agent), std::end(agents));

            auto newFactor = graph_.getFactor(agents);

            // Unfortunately here we cannot simply dump the new results in
            // the old factor as we do in the normal VariableElimination.
            // This is because in VariableElimination all elements are
            // summed together, which means that it doesn't matter whether
            // they are grouped or not. Here elements are CROSS-summed,
            // which means we cannot simply dump stuff lest losing a
            // cross-summing step.
            newFactor->getData().rules_ = mergePayoffs(std::move(newFactor->getData().rules_), std::move(newRules));
        }
    }

    bool ruleComp(const MOVE::Rule & lhs, const MOVE::Rule & rhs) {
        return veccmp(std::get<0>(lhs).second, std::get<0>(rhs).second) < 0;
    }

    MOVE::Rules mergePayoffs(MOVE::Rules && lhs, MOVE::Rules && rhs) {
        MOVE::Rules retval;
        // We're going to have at least these rules.
        retval.reserve(lhs.size() + rhs.size());

        std::sort(std::begin(lhs), std::end(lhs), ruleComp);
        std::sort(std::begin(rhs), std::end(rhs), ruleComp);

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


    std::vector<const MOVE::Entries*> getPayoffs(const MOVE::Rules & rules, const PartialAction & jointAction) {
        std::vector<const MOVE::Entries*> retval;
        // Note here that we must use match since the factors adjacent to
        // one agent aren't all next to all its neighbors. Since they are
        // different, we must coarsely check that equal agents do equal
        // actions.
        for (const auto & rule : rules)
            if (match(jointAction, std::get<0>(rule)))
                retval.push_back(&std::get<1>(rule));
        return retval;
    }

    MOVE::Entries crossSum(const MOVE::Entries & lhs, const std::vector<const MOVE::Entries*> rhs) {
        if (!rhs.size()) return lhs;

        MOVE::Entries retval;
        for (auto p : rhs) {
            auto tmp = crossSum(lhs, *p);
            retval.insert(std::end(retval), std::make_move_iterator(std::begin(tmp)),
                                            std::make_move_iterator(std::end(tmp)));
        }
        return retval;
    }

    MOVE::Entries crossSum(const MOVE::Entries & lhs, const MOVE::Entries & rhs) {
        if (!lhs.size()) return rhs;
        if (!rhs.size()) return lhs;
        MOVE::Entries retval;
        retval.reserve(lhs.size() + rhs.size());
        // We do the rhs last since they'll usually be shorter (due to
        // this class usage), so hopefully we can use the cache better.
        for (const auto & lhsVal : lhs) {
            for (const auto & rhsVal : rhs) {
                auto tags = merge(std::get<0>(lhsVal), std::get<0>(rhsVal));
                auto values = std::get<1>(lhsVal) + std::get<1>(rhsVal);
                retval.emplace_back(std::move(tags), std::move(values));
            }
        }
        return retval;
    }
}
