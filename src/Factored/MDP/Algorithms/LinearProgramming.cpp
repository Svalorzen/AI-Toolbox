#include <AIToolbox/Factored/MDP/Algorithms/LinearProgramming.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/LP.hpp>
#include <AIToolbox/Factored/Utils/GenericVariableElimination.hpp>

namespace AIToolbox::Factored::MDP {
    // Initial typedefs and definitions
    namespace {
        using Factor = size_t;
        using VE = GenericVariableElimination<Factor>;
        struct Global {
            LP & lp;

            Factor newFactor;

            void initNewFactor();
            void beginCrossSum();
            void crossSum(const Factor & f);
            void endCrossSum();
            void makeResult(VE::FinalFactors && finalFactors);
        };
    }

    std::tuple<Vector, QFunction> LinearProgramming::operator()(const CooperativeModel & m, const FactoredVector & h) const {
        std::tuple<Vector, QFunction> retval;
        auto & [v, g] = retval;

        g = backProject(m.getS(), m.getA(), m.getTransitionFunction(), h);
        auto values = solveLP(m, g, h);

        if (!values)
            throw std::runtime_error("Could not solve the LP for this MDP");

        v = std::move(*values);

        // Since we have already computed 'g', we compute Q as well.
        g *= m.getDiscount() * v;
        plusEqual(m.getS(), m.getA(), g, m.getRewardFunction());

        return retval;
    }

    std::optional<Vector> LinearProgramming::solveLP(const CooperativeModel & m, const FactoredMatrix2D & g, const FactoredVector & h) const {
        const auto & S = m.getS();
        const auto & A = m.getA();
        const auto & R = m.getRewardFunction();
        const auto discount = m.getDiscount();

        // Clear everything so we can use this function multiple times.
        VE::Graph graph(S.size() + A.size());

        // Normally, to solve an MDP via linear programming we have a series of
        // constraints for every possible s and a on the form of:
        //
        //     R(s,a) + discount * sum_s' T(s,a,s') * V*(s') <= V(s)
        //
        // Since we are working in factored form (both in states and actions),
        // we can convert these to:
        //
        //     R(s,a) + discount * sum_s' T(s,a,s') * [ sum_k w_k * h_k(s') ] <= sum_k w_k * h_k(s)
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
        // Thus, g is simply the FactoredMatrix2D returned from our
        // backProject() operator, but does NOT include the weights!
        //
        // Continuing from where we left off, we rewrite the constraints using
        // 'g' more compactly:
        //
        //     R(s,a) + discount * sum_k w_k * g_k(s,a) <= sum_k w_k * h_k(s)
        //     R(s,a) + sum_k w_k (discount * g_k(s,a) - h_k(s)) <= 0
        //
        // We can finally use the max operator to obtain a single constraint,
        // which will allow us to find the 'w' to best approximate the true V*:
        //
        //     max_s,a R(s,a) + sum_k w_k (discount * g_k(s,a) - h_k(s)) <= 0
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
        // - +w_k * (discount * g_k(s,a))
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

        const size_t returnVars = h.bases.size();

        // Since it is very very likely that our input `h` is composed by
        // indicator functions, where most of the elements are zeroes, we want
        // to avoid building zero rules as they just slow down the LP solve
        // process without adding anything. So we add a column at a time just
        // for the non-zero entries.
        LP lp(returnVars);

        for (size_t i = 0; i < h.bases.size(); ++i)
            lp.row[i] = h.bases[i].values.sum() / h.bases[i].values.size();
        lp.setObjective(false);

        // In this initial setup, we simply kind of give a "name"/"variable" to
        // each assignment of the inputs functions - note that all
        // operators are equalities here.
        //
        // This is not strictly necessary and could be optimized away, but for
        // now it is like this to make the removeState() code uniform as it
        // just needs to reference the constraints in the graph.

        // h setup: -w_k * (h_k(s))
        size_t currentWeight = 0; // This is basically k
        size_t currentRule = returnVars; // This is the "name" of the rule we are inserting in the graph

        for (const auto & f : h.bases) {
            auto newFactor = graph.getFactor(f.tag);
            for (int sId = 0; sId < f.values.size(); ++sId) {
                if (checkEqualSmall(f.values[sId], 0.0)) continue;
                // Add a column and re-initialize row
                lp.addColumn();
                lp.row.setZero();

                lp.row[currentRule] = -1.0; // Rule name for this value
                lp.row[currentWeight] = -f.values[sId];
                lp.pushRow(LP::Constraint::Equal, 0.0);

                newFactor->getData().emplace_back(sId, currentRule);
                currentRule += 1;
            }
            ++currentWeight;
        }

        // g setup: +w_k * (discount * g_k(s,a))
        currentWeight = 0;
        for (const auto & f : g.bases) {
            auto newFactor = graph.getFactor(join(S.size(), f.tag, f.actionTag));
            auto aMult = 1;
            for (auto id : f.tag) aMult *= S[id];
            for (int sId = 0; sId < f.values.rows(); ++sId) {
                for (int aId = 0; aId < f.values.cols(); ++aId) {
                    if (checkEqualSmall(f.values(sId, aId), 0.0)) continue;
                    // Add a column and re-initialize row
                    lp.addColumn();
                    lp.row.setZero();

                    lp.row[currentRule] = -1.0; // Rule name for this value
                    lp.row[currentWeight] = +discount * f.values(sId, aId);
                    lp.pushRow(LP::Constraint::Equal, 0.0);

                    newFactor->getData().emplace_back(sId + aMult * aId, currentRule);
                    currentRule += 1;
                }
            }
            ++currentWeight;
        }

        // R setup: +R(s,a)
        for (const auto & f : R.bases) {
            auto newFactor = graph.getFactor(join(S.size(), f.tag, f.actionTag));
            auto aMult = 1;
            for (auto id : f.tag) aMult *= S[id];
            for (int sId = 0; sId < f.values.rows(); ++sId) {
                for (int aId = 0; aId < f.values.cols(); ++aId) {
                    if (checkEqualSmall(f.values(sId, aId), 0.0)) continue;
                    // Add a column and re-initialize row
                    lp.addColumn();
                    lp.row.setZero();

                    lp.row[currentRule] = +1.0; // Rule name for this value
                    lp.pushRow(LP::Constraint::Equal, f.values(sId, aId));

                    newFactor->getData().emplace_back(sId + aMult * aId, currentRule);
                    currentRule += 1;
                }
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
        VE ve;
        Global global{lp, 0};

        const auto F = join(S, A);
        ve(F, graph, global);

        // Set every variable we have added as unbounded, since they shouldn't
        // be limited to positive (which may be the default).
        for (int i = 0; i < lp.row.size(); ++i)
            lp.setUnbounded(i);

        // Finally, try to solve the LP and find out the coefficients to
        // approximate V* using h!
        return lp.solve(returnVars);
    }

    void Global::initNewFactor() {
        newFactor = lp.row.size();

        lp.addColumn();
    }

    void Global::beginCrossSum() {
        // Each cross-sum creates a
        //
        //     newFactor >= sum of other_constraints
        //
        // rule in the LP, so we start the setup here.
        lp.row.setZero();
        lp.row[newFactor] = -1.0;
    }

    void Global::crossSum(const Factor & f) {
        // We add to the constraint all the factors to add.
        lp.row[f] = 1.0;
    }

    void Global::endCrossSum() {
        // We finally add the constraint we have setup to the LP.
        lp.pushRow(LP::Constraint::LessEqual, 0.0);
    }

    void Global::makeResult(VE::FinalFactors && finalFactors) {
        // Finally, add the last inequalities for all remaining factors.
        lp.row.setZero();

        for (const auto ruleId : finalFactors) {
            lp.row[ruleId] = 1.0;
            lp.pushRow(LP::Constraint::LessEqual, 0.0);
            lp.row[ruleId] = 0.0;
        }
    }
}
