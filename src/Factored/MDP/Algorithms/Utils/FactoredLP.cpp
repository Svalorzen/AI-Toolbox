#include <AIToolbox/Factored/MDP/Algorithms/Utils/FactoredLP.hpp>

#include <AIToolbox/LP.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    // Optimizations TODO:
    //     Use sparse pushRow to add rows.
    //     Add multiple columns at the same time.
    //     Reserve memory in advance for both rows and cols.
    //     Remove initial variables - "paste" them in.

    std::optional<Vector> FactoredLP::operator()(const FactoredVector & C, const FactoredVector & b) {
        // Clear everything so we can use this function multiple times.
        Graph graph(S.size());
        std::vector<size_t> finalFactors;

        // C = set of basis functions
        // B = set of target functions

        // So what we want to do here is to find the weights that minimize the
        // max-norm difference over all states between Cw and b.
        //
        // We are going to do this using an LP, which is built following the
        // steps of a variable elimination algorithm.
        //
        // This is because the key idea here is to maximize:
        //
        //     phi >= | (Cw)_i - b_i |
        //
        // Which is equivalent to
        //
        //     phi >= max_i | (Cw)_i - b_i |
        //
        // And finally, as we actually do it here (with phi minimized):
        //
        //    max_i | (Cw)_i - b_i | - phi <= 0
        //
        // Thus, our LP construct will be built so that the constraints are
        // basically going to refer to that max row where the difference is
        // highest, and the LP is then going to try to squish that value to the
        // lowest possible.
        //
        // Building this LP will require adding an unknown number of columns
        // (in particular, two per each VE rule, as we want to maximize over an
        // absolute value, so we do it "forward and backward").

        const auto phiId = C.bases.size(); // Skip ws since we want to extract those later.
        size_t startingVars = phiId + 1;   // ws + phi
        for (const auto & f : C.bases) startingVars += f.values.size() * 2;
        for (const auto & f : b.bases) startingVars += f.values.size() * 2;

        // Init LP with starting variables
        LP lp(startingVars);
        lp.setObjective(phiId, false); // Minimize phi
        lp.row.fill(0.0);

        // In this initial setup, we simply kind of give a "name"/"variable" to
        // each assignment of the original Cw and b functions - note that all
        // operators are equalities here. In the first loop we do C (which is
        // thus associated with the weights), and in the second b (which is
        // not).
        //
        // This is not strictly necessary and could be optimized away, but for
        // now it is like this to make the removeState() code uniform as it
        // just needs to reference the constraints in the graph.
        size_t wi = 0;
        size_t currentRule = phiId + 1; // Skip ws + phi
        for (const auto & f : C.bases) {
            auto newFactor = graph.getFactor(f.tag);
            for (size_t i = 0; i < static_cast<size_t>(f.values.size()); ++i) {
                lp.row[currentRule] = -1.0;
                lp.row[wi] = f.values[i];
                lp.pushRow(LP::Constraint::Equal, 0.0);
                lp.row[currentRule] = 0.0;

                lp.row[currentRule+1] = -1.0;
                lp.row[wi] = -f.values[i];
                lp.pushRow(LP::Constraint::Equal, 0.0);
                lp.row[currentRule+1] = 0.0;

                newFactor->getData().emplace_back(std::make_pair(f.tag, toFactorsPartial(f.tag, S, i)), currentRule);
                currentRule += 2;
            }
            lp.row[wi++] = 0.0;
        }
        // Here signs are opposite to those of C since we need to find (Cw - b)
        // and (b - Cw)
        for (const auto & f : b.bases) {
            auto newFactor = graph.getFactor(f.tag);
            for (size_t i = 0; i < static_cast<size_t>(f.values.size()); ++i) {
                lp.row[currentRule] = 1.0;
                lp.pushRow(LP::Constraint::Equal, -f.values[i]);
                lp.row[currentRule] = 0.0;

                lp.row[currentRule+1] = 1.0;
                lp.pushRow(LP::Constraint::Equal, f.values[i]);
                lp.row[currentRule+1] = 0.0;

                newFactor->getData().emplace_back(std::make_pair(f.tag, toFactorsPartial(f.tag, S, i)), currentRule);
                currentRule += 2;
            }
        }

        // Now that the setup is done and we have internalized the inputs as
        // rules, we now create the rest mirroring the steps in variable
        // elimination.
        //
        // To do this, we basically eliminate a part of the state space (which
        // is a vector) at a time, maximizing on one of the State components at
        // a time. We don't really do anything here aside from creating new
        // constraints in the LP, and giving them "names".
        while (graph.variableSize())
            removeState(graph, graph.variableSize() - 1, lp, finalFactors);

        // Finally, add the two phi rules for all remaining factors.
        lp.row.fill(0.0);
        lp.row[phiId] = -1.0;

        for (const auto ruleId : finalFactors)
            lp.row[ruleId] = 1.0;

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

    void FactoredLP::removeState(Graph & graph, size_t s, LP & lp, std::vector<size_t> & finalFactors) {
        const auto factors = graph.getNeighbors(s);
        auto variables = graph.getNeighbors(factors);

        PartialFactorsEnumerator jointActions(S, variables, s);
        const auto id = jointActions.getFactorToSkipId();
        Rules newRules;

        // We'll now create new rules that represent the elimination of the
        // input variable for this round. For each possible assignment to the
        // variables, we create two rules: one for (Cw - b) and one for (b -
        // Cw).
        const bool isFinalFactor = variables.size() == 1;

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

            if (!isFinalFactor)
                newRules.emplace_back(*jointActions, newRuleId);
            else
                finalFactors.push_back(newRuleId);

            jointActions.advance();
        }

        // And finally as usual in variable elimination remove the variable
        // from the graph and insert the newly created variable in.

        for (const auto & it : factors)
            graph.erase(it);
        graph.erase(s);

        if (!isFinalFactor) {
            variables.erase(std::remove(std::begin(variables), std::end(variables), s), std::end(variables));

            auto newFactor = graph.getFactor(variables);
            newFactor->getData().insert(
                    std::end(newFactor->getData()),
                    std::make_move_iterator(std::begin(newRules)),
                    std::make_move_iterator(std::end(newRules))
            );
        }
    }
}
