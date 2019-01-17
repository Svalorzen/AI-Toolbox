#ifndef AI_TOOLBOX_FACTORED_MDP_FACTORED_VALUE_ITERATION_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_FACTORED_VALUE_ITERATION_HEADER_FILE

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>
#include <AIToolbox/LP.hpp>

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

    // Since here the parents of bf can be actions (since we are
    // backpropagating in a DDN rather than a DBN), the output must also be a
    // 2d matrix. Worst case we can consider states and actions together, but
    // I'd prefer not to.
   BasisMatrix backProject(const Factors & space, const Factors & actions, const FactoredDDN & ddn, const BasisFunction & bf) {
       BasisMatrix retval;

        for (auto d : bf.tag) {
            retval.actionTag = merge(retval.actionTag, ddn[d].actionTag);
            for (const auto & n : ddn[d].nodes)
                retval.tag = merge(retval.tag, n.tag);
        }

        const size_t sizeA = factorSpacePartial(retval.actionTag, actions);
        const size_t sizeS = factorSpacePartial(retval.tag, space);

        retval.values.resize(sizeS, sizeA);

        size_t sId = 0;
        size_t aId = 0;

        PartialFactorsEnumerator sDomain(space, retval.tag);
        PartialFactorsEnumerator aDomain(actions, retval.actionTag);

        PartialFactorsEnumerator rhsDomain(space, bf.tag);

        while (sDomain.isValid()) {
            while (aDomain.isValid()) {
                // For each domain assignment, we need to go over every
                // possible children assignment. As we are computing
                // products, it is sufficient to go over the elements
                // stored in the RHS (as all other children combinations
                // are zero by definition).
                //
                // For each such assignment, we compute the product of the
                // rhs there with the value of the lhs at the current
                // domain & children.
                double currentVal = 0.0;
                size_t i = 0;
                while (rhsDomain.isValid()) {
                    currentVal += bf.values[i] * ddn.getTransitionProbability(space, actions, *sDomain, *aDomain, *rhsDomain);

                    ++i;
                    rhsDomain.advance();
                }
                retval.values(sId, aId) = currentVal;

                ++aId;
                aDomain.advance();
                rhsDomain.reset();
            }
            ++sId;
            sDomain.advance();
        }
        return retval;
    }

    // Performs the bellman equation on a single action
    template <typename BN>
    inline FactoredVector bellmanEquation(const State & S, double discount, const BN & T, const FactoredVector & A, const Vector & w, const FactoredVector & R) {
        // Q = R + gamma * T * (A * w)
        FactoredVector Q = backProject(S, T, A * w);
        Q *= discount;
        return plusEqual(S, Q, R);
    }

    // Performs the bellman equation on a single action
    inline Factored2DMatrix bellmanEquation(const State & S, const Action & A, double discount, const FactoredDDN & T, const FactoredVector & V, const Vector & w, const Factored2DMatrix & R) {
        // Q = R + gamma * T * (A * w)
        Factored2DMatrix Q = backProject(S, A, T, V * w);
        Q *= discount;
        return plusEqual(S, A, Q, R);
    }

    std::optional<Vector> test(const Factored2DMatrix & R, const Factored2DMatrix & g, const FactoredVector & h, bool addConstantBasis) {
        // Clear everything so we can use this function multiple times.
        using Graph = FactorGraph<int>;
        Graph graph(1000);
        std::vector<size_t> finalFactors;

        // C = set of basis functions
        // B = set of target functions

        // Normally, to solve an MDP via linear programming we have a series of
        // constraints for every possible s and a on the form of:
        //
        //     R(s,a) + gamma * sum_s' T(s,a,s') * V*(s') <= V(s)
        //
        // Since we are working in factored form (both in states and actions),
        // we can convert these to:
        //
        //     R(s,a) + gamma * sum_s' T(s,a,s') * [ sum_k w_k * h_k(s') ] <= sum_k w_k * h_k(s)
        //
        // Where the 'h's stand for the basis functions that compose our
        // factored V function. Note that T is instead the Dynamic Decision
        // Network of our factored MDP.
        //
        // Before continuing, for simplicity of exposition, we're going to
        // introduce a new function 'g', equal to:
        //
        //     g(s,a) = sum_s' T(s,a,s') * h_k(s')
        //
        // Thus, g is simply the Factored2DMatrix returned from our
        // backProject() operator, but does NOT include the weights!
        //
        // Continuing from where we left off, we rewrite the constraints using
        // 'g' more compactly:
        //
        //     R(s,a) + gamma * sum_k w_k * g_k(s,a) <= sum_k w_k * h_k(s)
        //     R(s,a) + sum_k w_k (gamma * g_k(s,a) - h_k(s)) <= 0
        //
        // We can finally use the max operator to obtain a single constraint,
        // which will allow us to find the 'w' to best approximate the true V*:
        //
        //     max_s,a R(s,a) + sum_k w_k (gamma * g_k(s,a) - h_k(s)) <= 0
        //
        // As per the MDP::LinearProgramming class, we're trying to minimize:
        //
        //     minimize sum_s 1/S * V(s)
        //
        // Where the 1/S is the importance coefficient of a state, and here we
        // consider them all equally. In the factored case, this becomes:
        //
        //     minimize sum_k 1/|h_k| * w_k
        //
        // So we actually want to minimize the weights.
        //
        // So now we need to create an LP which follows the construction of the
        // FactoredLP class: our variables are S and A combined, and our
        // functions are R, g and h.
        //
        // So we're going to add the following variables to the LP (for every
        // possible s,a combination):
        //
        // - +R(s,a) # Note that R is also factorized
        // - +w_k * (gamma * g_k(s,a))
        // - -w_k * (h_k(s))
        //
        // Afterwards, we're just going to run VE and return the best w.

        // NOTE: The removeState function from factoredLP can probably be reused.
        //
        // The question is: can we alter the API to build arbitrary functions?
        // Could be done, like: some overloads of "addRules(matrix, +/-, useweights, addConstantBasis)"
        // Then one can build whatever they want? But then one needs to be able to set the target as well.
        //
        // Leave this for later after seeing how this looks.

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

    // inline void makePolicy(const State & S, const CompactDDN & p, const std::vector<FactoredVector> & q, const FactoredVector & A, const Vector & w, const FactoredVector & R) {
    //     std::vector<std::tuple<PartialState, size_t, double>> retval;

    //     auto defaultQ = backProject(S, p.getDefaultTransition(), A * w);
    //     // defaultQ *= discount;

    //     const auto & x = p.getDiffNodes();

    //     for (size_t a = 0; a < q.size(); ++a) {
    //         std::vector<size_t> Ia;
    //         // Compute the bases for which we actually need to compare in order
    //         // to determine the delta of this action w.r.t. the default BN.
    //         for (size_t i = 0; i < A.bases.size(); ++i) {
    //             for (size_t j = 0; j < x[a].size(); ++j) {
    //                 if (sequential_sorted_contains(A.bases[i].tag, x[a][j].id)) {
    //                     Ia.push_back(i);
    //                     break;
    //                 }
    //             }
    //         }
    //     }
    // }
}

#endif
