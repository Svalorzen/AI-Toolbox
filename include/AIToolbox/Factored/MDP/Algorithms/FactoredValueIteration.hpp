#ifndef AI_TOOLBOX_FACTORED_MDP_FACTORED_VALUE_ITERATION_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_FACTORED_VALUE_ITERATION_HEADER_FILE

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace AIToolbox::Factored::MDP {
    // TODO:
    //
    // So in order to make the thing work, let's say again what we need to do:
    //
    // - We need to define a factored matrix which can contain both R and
    //   Q, and that is possibly easily convertible to the Q form we have
    //   already been using (FactoredContainer<QFunctionRule>).
    //   ! It might make sense to really use that directly, as I don't think
    //   there;s any access pattern there..
    //   MMh. Note that we want the end result to be factored: Q = sum_j Q_j
    //   where each Q_j only has a small domain (which also allows us to easily
    //   sum R, since we just append it).
    //   So when we do VE we probably want to build the QFunctionRules on the
    //   fly based on what matches.. where we do the matches on the tags.
    // - We need to define a backpropagation for it. This needs to work also
    //   with the new FactoredDDN.
    // - We need to make the LP work with these things.
    //
    // Everything should work then.
    //
    // BONUS - Think how to make QGreedyPolicy work with both
    // FactoredContainer<QFunctionRule> and a Factored2DMatrix!

    // Rename to Factored2DMatrix, since the other one at this point does not
    // count as it does not factorize actions. Add this as a comment on the
    // structure, however it is made.
    //
    // Since backproject seems to return *dense* matrices over a tag e
    // actiontag, maybe an equivalent basisfunction could be:
    //
    // PartialKeys tag;
    // PartialKeys actionTag;
    // Matrix2D values;
    //
    // And from THERE you get a Factored2DMatrix. Seems decent enough, and
    // mirrors the FactoredVector which is cool.
    using UltraFactored2DMatrix = int;

    // Since here the parents of bf can be actions (since we are
    // backpropagating in a DDN rather than a DBN), the output must also be a
    // 2d matrix. Worst case we can consider states and actions together, but
    // I'd prefer not to.
    UltraFactored2DMatrix backProject(const Factors & space, const FactoredDDN & dbn, const BasisFunction & bf) {
        // Domain of retval = for each d in bf.tag:
        //                        retval.action_domain += dbn[d].actionTag;
        //                        for node in dbn[d].nodes:
        //                            retval.state_domain += node.tag;
        //
        // For every combA in retval.action_domain:
        // For every combS in retval.state_domain:
        //     V = 0.0
        //     For every combS1 in bf.tag:
        //         V += bf.getValue(combS1) * dbn.getTransitionProbability(space, actions, combS, combA, combS1);
        //
        //     retval[combS][combA] = V;
        //
        // return retval;
    }

    // Performs the bellman equation on a single action
    template <typename BN>
    inline FactoredVector bellmanEquation(const State & S, double discount, const BN & T, const FactoredVector & A, const Vector & w, const FactoredVector & R) {
        // Q = R + gamma * T * (A * w)
        FactoredVector Q = backProject(S, T, A * w);
        Q *= discount;
        return plusEqual(S, Q, R);
    }

    inline void makePolicy(const State & S, const CompactDDN & p, const Factored2DMatrix & q, const FactoredVector & A, const Vector & w, const FactoredVector & R) {
        std::vector<std::tuple<PartialState, size_t, double>> retval;

        auto defaultQ = backProject(S, p.getDefaultTransition(), A * w);
        // defaultQ *= discount;

        const auto & x = p.getDiffNodes();

        for (size_t a = 0; a < q.size(); ++a) {
            std::vector<size_t> Ia;
            // Compute the bases for which we actually need to compare in order
            // to determine the delta of this action w.r.t. the default BN.
            for (size_t i = 0; i < A.bases.size(); ++i) {
                for (size_t j = 0; j < x[a].size(); ++j) {
                    if (sequential_sorted_contains(A.bases[i].tag, x[a][j].id)) {
                        Ia.push_back(i);
                        break;
                    }
                }
            }
        }
    }

    std::optional<Vector> test()(const FactoredVector & C, const FactoredVector & b, bool addConstantBasis) {
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

        const auto phiId = C.bases.size() + (addConstantBasis); // Skip ws since we want to extract those later.

        size_t startingVars = phiId + 1;   // ws + phi
        for (const auto & f : C.bases) startingVars += f.values.size() * 2;
        for (const auto & f : b.bases) startingVars += f.values.size() * 2;

        // Init LP with starting variables
        LP lp(startingVars);
        lp.setObjective(phiId, false); // Minimize phi
        lp.row.fill(0.0);

        // Compute constant basis useful values (only used if needed)
        const auto constBasisId = phiId - 1;
        const double constBasisCoeff = 1.0 / C.bases.size();

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
                if (addConstantBasis) lp.row[constBasisId] = constBasisCoeff;
                lp.pushRow(LP::Constraint::Equal, 0.0);
                lp.row[currentRule] = 0.0;

                lp.row[currentRule+1] = -1.0;
                lp.row[wi] = -f.values[i];
                if (addConstantBasis) lp.row[constBasisId] = -constBasisCoeff;
                lp.pushRow(LP::Constraint::Equal, 0.0);
                lp.row[currentRule+1] = 0.0;

                newFactor->getData().emplace_back(std::make_pair(f.tag, toFactorsPartial(f.tag, S, i)), currentRule);
                currentRule += 2;
            }
            lp.row[wi++] = 0.0;
        }
        lp.row[constBasisId] = 0.0;

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
}

#endif
