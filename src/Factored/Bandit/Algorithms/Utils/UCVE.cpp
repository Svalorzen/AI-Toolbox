#include <AIToolbox/Factored/Bandit/Algorithms/Utils/UCVE.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Prune.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Impl/Logging.hpp>

#include <boost/iterator/transform_iterator.hpp>

namespace AIToolbox::Factored::Bandit {
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
    UCVE::Entries crossSum(const UCVE::Entries & lhs, const UCVE::Entries & rhs);

    /**
     * @brief This function cross-sums the input lists.
     *
     * Cross-sums are performed considering the right-hand side as one
     * single joined list. This is useful considering how the getPayoffs()
     * function works.
     *
     * \sa crossSum(const UCVE::Entries &, const UCVE::Entries &);
     *
     * @param lhs The left hand side.
     * @param rhs A list of pointers to valid Entries lists.
     *
     * @return A new list containing all cross-sums.
     */
    UCVE::Entries crossSum(const UCVE::Entries & lhs, const std::vector<const UCVE::Entries*> rhs);

    /**
     * @brief This function returns a list of pointers to all Entries from the Rules matching the input joint action.
     *
     * @param keys The keys of the agents of the rules.
     * @param rules A list of Rule.
     * @param jointAction A joint action to match Rules against.
     *
     * @return A list of pointers to the Entries contained in the Rules matched against the input action.
     */
    std::vector<const UCVE::Entries*> getPayoffs(const PartialKeys & keys, const UCVE::Rules & rules, const PartialAction & jointAction);

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
    UCVE::Rules mergePayoffs(UCVE::Rules && lhs, UCVE::Rules && rhs);

    // We half the logtA since we always need to multiply it with 1/2 anyway.
    UCVE::UCVE(Action a, double logtA) : A(std::move(a)), graph_(A.size()), logtA_(logtA * 0.5) {}

    // We use this to compute the UCB value given a bound
    double computeValue(const UCVE::Entry & e, const double x, const double logtA);

    UCVE::Result UCVE::start() {
        // This can possibly be improved with some heuristic ordering
        while (graph_.variableSize())
            removeAgent(graph_.variableSize() - 1);

        AI_LOGGER(AI_SEVERITY_DEBUG, "Done removing agents.");
        if (finalFactors_.size() == 0) return {};

        AI_LOGGER(AI_SEVERITY_DEBUG, "Picking best final factors...");
        Result retval; std::get<1>(retval).fill(0.0);
        for (const auto & fValue : finalFactors_) {
            const auto begin = fValue.begin(), end = fValue.end();

            double max = computeValue(*begin, 0.0, logtA_);
            auto maxIt = begin;
            for (auto it = begin + 1; it != end; ++it) {
                const auto tmp = computeValue(*it, 0.0, logtA_);
                if (tmp > max) {
                    max = tmp;
                    maxIt = it;
                }
            }
            std::get<0>(retval) = merge(std::get<0>(retval), std::get<0>(*maxIt));
            std::get<1>(retval) += std::get<1>(*maxIt);
        }
        return retval;
    }

    void UCVE::removeAgent(const size_t agent) {
        AI_LOGGER(AI_SEVERITY_DEBUG, "Removing agent " << agent);

        const auto factors = graph_.getNeighbors(agent);
        auto agents = graph_.getNeighbors(factors);

        const bool isFinalFactor = agents.size() == 1;

        AI_LOGGER(AI_SEVERITY_DEBUG, "This agent has " << factors.size() << " factors.");
        AI_LOGGER(AI_SEVERITY_DEBUG, "Now building bounds...");

        // Here we compute the upper and lower bounds for this round. These
        // are used later for pruning, since here each V we get is
        // basically just a mean estimate plus a variance to guess just how
        // much the rule could actually be worth. We will want to prune
        // those that, in their best case, still do not come out better
        // then at least one in its worst case.
        double x_u = 0.0, x_l = 0.0;
        {
            // We use these iterators to skip the factors for this agent.
            auto skipIt = factors.cbegin(); const auto factorsEnd = factors.cend();
            for (auto it = graph_.cbegin(); it != graph_.cend(); ++it) {
                // We skip the ones for this agent. Both lists are in the same
                // order so we can keep track of the last duplicate we found to
                // do less work later.
                const auto duplicate = std::find(skipIt, factorsEnd, it);
                if (duplicate != factorsEnd) {
                    skipIt = duplicate + 1;
                    continue;
                }
                double currMax = std::numeric_limits<double>::lowest();
                double currMin = std::numeric_limits<double>::max();
                for (const auto & rule : it->getData().rules) {
                    for (const auto & entry : std::get<1>(rule)) {
                        const auto & v = std::get<1>(entry);
                        currMax = std::max(currMax, v[1]);
                        currMin = std::min(currMin, v[1]);
                    }
                }
                x_u += currMax;
                x_l += currMin;
            }
        }
        AI_LOGGER(AI_SEVERITY_DEBUG, "Current bounds: lower = " << x_l << "; higher = " << x_u);
        AI_LOGGER(AI_SEVERITY_DEBUG, "Cross-summing and pruning...");

        // Now that we are done computing the bounds, we perform the actual
        // cross-summing and related pruning. The pruning here uses the
        // bounds in order to do UCB and keep the most promising actions.
        Rules newRules;
        PartialFactorsEnumerator jointActions(A, agents, agent);
        const auto id = jointActions.getFactorToSkipId();
        while (jointActions.isValid()) {
            auto & jointAction = *jointActions;

            Entries values;
            for (size_t agentAction = 0; agentAction < A[agent]; ++agentAction) {
                jointAction.second[id] = agentAction;

                Entries newEntries;
                for (const auto p : getPayoffs(factors[0]->getVariables(), factors[0]->getData().rules, jointAction))
                    newEntries.insert(std::end(newEntries), std::begin(*p), std::end(*p));

                auto entries = newEntries.size();
                for (size_t i = 1; i < factors.size(); ++i) {
                    newEntries = crossSum(newEntries, getPayoffs(factors[i]->getVariables(), factors[i]->getData().rules, jointAction));
                    // We remove the entries that cannot possibly be useful anymore
                    if (newEntries.size() > entries) {
                        newEntries.erase(boundPrune(std::begin(newEntries), std::end(newEntries), x_l, x_u), std::end(newEntries));
                        entries = newEntries.size();
                    }
                }

                if (newEntries.size() != 0) {
                    AI_LOGGER(AI_SEVERITY_DEBUG, "Adding entries...");
                    // Add tags for the current agent.
                    for (auto & nv : newEntries) {
                        auto & first  = std::get<0>(nv).first;
                        auto & second = std::get<0>(nv).second;
                        // Find where the current agent should be.
                        size_t i = 0;
                        while (i < first.size() && first[i] < agent) ++i;
                        // Insert the agent and its action
                        first.insert(std::begin(first) + i, agent);
                        second.insert(std::begin(second) + i, agentAction);
                    }
                    values.insert(std::end(values), std::make_move_iterator(std::begin(newEntries)),
                                                    std::make_move_iterator(std::end(newEntries)));
                }
            }
            if (values.size() != 0) {
                // If this is a final factor we do the alternative path
                // here, to avoid copying joint actions which we won't
                // really need anymore.
                if (!isFinalFactor) {
                    AI_LOGGER(AI_SEVERITY_DEBUG, "Found new rule...");
                    newRules.emplace_back(PartialValues(), std::move(values));
                    newRules.back().first.reserve(agents.size() - 1);
                    for (size_t a = 0; a < agents.size(); ++a)
                        if (a != id) newRules.back().first.push_back(jointAction.second[a]);
                } else {
                    AI_LOGGER(AI_SEVERITY_DEBUG, "Adding final factor...");
                    finalFactors_.emplace_back(std::move(values));
                }
            }
            jointActions.advance();
        }
        AI_LOGGER(AI_SEVERITY_DEBUG, "Done. Erasing agent...");

        for (const auto & it : factors)
            graph_.erase(it);
        graph_.erase(agent);

        AI_LOGGER(AI_SEVERITY_DEBUG, "Done.");

        if (newRules.size() == 0) return;
        if (!isFinalFactor) {
            AI_LOGGER(AI_SEVERITY_DEBUG, "Non-end rule, adding it...");
            agents.erase(std::remove(std::begin(agents), std::end(agents), agent), std::end(agents));

            auto newFactor = graph_.getFactor(agents);
            auto & fRules = newFactor->getData().rules;

            // Unfortunately here we cannot simply dump the new results in
            // the old factor as we do in the normal VariableElimination.
            // This is because in VariableElimination all elements are
            // summed together, which means that it doesn't matter whether
            // they are grouped or not. Here elements are CROSS-summed,
            // which means we cannot simply dump stuff lest losing a
            // cross-summing step.
            fRules = mergePayoffs(std::move(fRules), std::move(newRules));
        }
    }

    double computeValue(const UCVE::Entry & e, const double x, const double logtA) {
        // Note: the 1/2 is implied in logtA
        return std::get<1>(e)[0] + std::sqrt((std::get<1>(e)[1] + x) * logtA);
    };

    UCVE::Entries::iterator UCVE::boundPrune(const Entries::iterator begin, Entries::iterator end, const double x_l, const double x_u) {
        if ( std::distance(begin, end) < 2 ) return end;

        // We first eliminate all dominated vectors, then we remove all that
        // can't possibly be useful using the bounds we have computed.
        const auto unwrap = +[](UCVE::Entry & entry) -> UCVE::V & {return std::get<1>(entry);};
        const auto rbegin = boost::make_transform_iterator(begin, unwrap);
        const auto rend   = boost::make_transform_iterator(end, unwrap);

        end = extractDominated(2, rbegin, rend).base();

        double max = computeValue(*begin, x_l, logtA_);
        auto maxIt = begin;
        for (auto it = begin + 1; it != end; ++it) {
            const auto tmp = computeValue(*it, x_l, logtA_);
            if (tmp > max) {
                max = tmp;
                maxIt = it;
            }
        }
        // Put the best first so we can use <= for the pruning (otherwise if we
        // didn't know we would be forced to use < to avoid removing the best)
        iter_swap(begin, maxIt);
        return std::remove_if(begin + 1, end, [max, x_u, logtA = logtA_](const UCVE::Entry & e) { return computeValue(e, x_u, logtA) <= max; });
    }

    std::vector<const UCVE::Entries*> getPayoffs(const PartialKeys & keys, const UCVE::Rules & rules, const PartialAction & jointAction) {
        std::vector<const UCVE::Entries*> retval;
        // Note here that we must use match since the factors adjacent to
        // one agent aren't all next to all its neighbors. Since they are
        // different, we must coarsely check that equal agents do equal
        // actions.
        for (const auto & rule : rules)
            if (match(keys, rule.first, jointAction.first, jointAction.second))
                retval.push_back(&rule.second);
        return retval;
    }

    UCVE::Entries crossSum(const UCVE::Entries & lhs, const std::vector<const UCVE::Entries*> rhs) {
        if (!rhs.size()) return lhs;

        UCVE::Entries retval;
        for (auto p : rhs) {
            auto tmp = crossSum(lhs, *p);
            retval.insert(std::end(retval), std::make_move_iterator(std::begin(tmp)),
                                            std::make_move_iterator(std::end(tmp)));
        }
        return retval;
    }

    UCVE::Entries crossSum(const UCVE::Entries & lhs, const UCVE::Entries & rhs) {
        if (!lhs.size()) return rhs;
        if (!rhs.size()) return lhs;
        UCVE::Entries retval;
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

    bool ruleComp(const UCVE::Rule & lhs, const UCVE::Rule & rhs) {
        return veccmp(lhs.first, rhs.first) < 0;
    }

    UCVE::Rules mergePayoffs(UCVE::Rules && lhs, UCVE::Rules && rhs) {
        UCVE::Rules retval;
        // We're going to have at least these rules.
        retval.reserve(lhs.size() + rhs.size());

        std::sort(std::begin(lhs), std::end(lhs), ruleComp);
        std::sort(std::begin(rhs), std::end(rhs), ruleComp);

        // Here we merge two lists of Rules. What we want is that if any of
        // them match, we need to crossSum them. Otherwise, just bring them
        // over to the result list unchanged.
        size_t i = 0, j = 0;
        while (i < lhs.size() && j < rhs.size()) {
            auto first = veccmp(lhs[i].first, rhs[j].first);
            if (first < 0)
                retval.emplace_back(std::move(lhs[i++]));
            else if (first > 0)
                retval.emplace_back(std::move(rhs[j++]));
            else {
                retval.emplace_back(lhs[i].first, crossSum(lhs[i].second, rhs[j].second));
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
}
