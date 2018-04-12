#include <AIToolbox/Factored/MDP/Algorithms/Utils/FactoredLP.hpp>

#include <AIToolbox/LP.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    // Optimizations TODO:
    //     Use sparse pushRow to add rows.
    //     Add multiple columns at the same time.
    //     Remove initial variables - "paste" them in.

    std::optional<Vector> FactoredLP::operator()(const FactoredFunction & C, const FactoredFunction & b) {
        // C = set of basis functions
        // B = set of target functions

        const auto phiId = C.factorSize(); // Skip ws since we want to extract those later.
        size_t startingVars = phiId + 1;   // ws + phi
        for (const auto & f : C) startingVars += f.getData().size() * 2;
        for (const auto & f : b) startingVars += f.getData().size() * 2;

        // Init LP with starting variables
        LP lp(startingVars);
        lp.setObjective(phiId, false); // Minimize phi
        lp.row.fill(0.0);

        size_t wi = 0;
        size_t currentRule = phiId + 1; // Skip ws + phi
        for (auto f = C.begin(); f != C.end(); ++f) {
            auto variables = C.getNeighbors(f);
            auto newFactor = graph_.getFactor(variables);
            for (const auto & entry : f->getData()) {
                lp.row[currentRule] = -1.0;
                lp.row[wi] = entry.value_;
                lp.pushRow(LP::Constraint::Equal, 0.0);
                lp.row[currentRule] = 0.0;

                lp.row[currentRule+1] = -1.0;
                lp.row[wi] = -entry.value_;
                lp.pushRow(LP::Constraint::Equal, 0.0);
                lp.row[currentRule+1] = 0.0;

                newFactor->getData().emplace_back(entry.s_, currentRule);
                currentRule += 2;
            }
            lp.row[wi++] = 0.0;
        }
        // Here signs are opposite to those of C since we need to find (Cw - b)
        // and (b - Cw)
        for (auto f = b.begin(); f != b.end(); ++f) {
            auto variables = b.getNeighbors(f);
            auto newFactor = graph_.getFactor(variables);
            for (const auto & entry : f->getData()) {
                lp.row[currentRule] = 1.0;
                lp.pushRow(LP::Constraint::Equal, -entry.value_);
                lp.row[currentRule] = 0.0;

                lp.row[currentRule+1] = 1.0;
                lp.pushRow(LP::Constraint::Equal, entry.value_);
                lp.row[currentRule+1] = 0.0;

                newFactor->getData().emplace_back(entry.s_, currentRule);
                currentRule += 2;
            }
        }

        // Now that the setup is done and we have internalized the inputs as
        // rules, we now create the rest mirroring the steps in variable
        // elimination.
        while (graph_.variableSize())
            removeState(graph_.variableSize() - 1, lp);

        // Finally, add the two phi rules for all remaining factors.
        lp.row.fill(0.0);
        lp.row[phiId] = -1.0;

        for (const auto ruleIds : finalFactors_)
            for (const auto ruleId : ruleIds)
                lp.row[std::get<1>(ruleId)] = 1.0;

        lp.pushRow(LP::Constraint::LessEqual, 0.0);

        // Now do the reverse for all opposite rules (same rules +1)
        for (size_t i = lp.row.size() - 2; i > phiId; --i) {
            if (lp.row[i] != 0.0) {
                lp.row[i+1] = lp.row[i];
                lp.row[i] = 0.0;
            }
        }

        lp.pushRow(LP::Constraint::LessEqual, 0.0);

        // Set every variable we have added as unbounded, since they shouldn't
        // be limited to positive (which may be the default).
        for (int i = 0; i < lp.row.size(); ++i)
            lp.setUnbounded(i);

        // Finally, try to solve the LP and find out the coefficients to
        // approximate b using C!
        return lp.solve(phiId);
    }

    void FactoredLP::removeState(size_t s, LP & lp) {
        const auto factors = graph_.getNeighbors(s);
        auto variables = graph_.getNeighbors(factors);

        PartialFactorsEnumerator jointActions(S, variables, s);
        const auto id = jointActions.getFactorToSkipId();
        Rules newRules;

        // We'll now create new rules that represent the elimination of the
        // input variable for this round. For each possible assignment to the
        // variables, we create two rules: one for (Cw - b) and one for (b -
        // Cw).

        while (jointActions.isValid()) {
            auto & jointAction = *jointActions;
            lp.addColumn();
            lp.addColumn();

            const size_t newRuleId = lp.row.size() - 2;

            for (size_t sAction = 0; sAction < S[s]; ++sAction) {
                lp.row.fill(0.0);
                lp.row[newRuleId] = -1.0;

                jointAction.second[id] = sAction;
                for (const auto ruleIds : factors)
                    for (const auto ruleId : ruleIds->getData())
                        if (match(jointAction, std::get<0>(ruleId)))
                            lp.row[std::get<1>(ruleId)] = 1.0;

                lp.pushRow(LP::Constraint::LessEqual, 0.0);

                // Now do the reverse for all opposite rules (same rules +1)
                for (int i = lp.row.size() - 2; i >= 0; --i) {
                    if (lp.row[i] != 0.0) {
                        lp.row[i+1] = lp.row[i];
                        lp.row[i] = 0.0;
                    }
                }
                lp.pushRow(LP::Constraint::LessEqual, 0.0);
            }

            newRules.emplace_back(*jointActions, newRuleId);
            jointActions.advance();
        }

        // And finally as usual in variable elimination remove the variable
        // from the graph and insert the newly created variable in.

        for (const auto & it : factors)
            graph_.erase(it);
        graph_.erase(s);

        if (variables.size() > 1) {
            variables.erase(std::remove(std::begin(variables), std::end(variables), s), std::end(variables));

            auto newFactor = graph_.getFactor(variables);
            newFactor->getData().insert(
                    std::end(newFactor->getData()),
                    std::make_move_iterator(std::begin(newRules)),
                    std::make_move_iterator(std::end(newRules))
            );
        } else {
            finalFactors_.push_back(newRules);
        }
    }
}
