#include <AIToolbox/Factored/MDP/Algorithms/Utils/FactoredLP.hpp>

#include <AIToolbox/Utils/LP.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/GenericVariableElimination.hpp>

namespace AIToolbox::Factored::MDP {
    // Initial typedefs and definitions
    namespace {
        using Factor = size_t;
        using VE = GenericVariableElimination<Factor>;
        struct Global {
            LP & lp;
            const size_t phiId;

            Factor newFactor;

            void initNewFactor();
            void beginCrossSum();
            void crossSum(const Factor & f);
            void endCrossSum();
            void makeResult(VE::FinalFactors && finalFactors);
        };
    }

    // Optimizations TODO:
    //     Use sparse pushRow to add rows.
    //     Add multiple columns at the same time.
    //     Reserve memory in advance for both rows and cols.
    //     Remove initial variables - "paste" them in.

    std::optional<Vector> FactoredLP::operator()(const FactoredVector & C, const FactoredVector & b, bool addConstantBasis) {
        // Clear everything so we can use this function multiple times.
        VE::Graph graph(S.size());

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
        lp.row.setZero();

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
        size_t currentWeight = 0;
        size_t currentRule = phiId + 1; // Skip ws + phi
        for (const auto & f : C.bases) {
            auto newFactor = graph.getFactor(f.tag);

            for (int i = 0; i < f.values.size(); ++i) {
                lp.row[currentRule] = -1.0;
                lp.row[currentWeight] = f.values[i];
                if (addConstantBasis) lp.row[constBasisId] = constBasisCoeff;
                lp.pushRow(LP::Constraint::Equal, 0.0);
                lp.row[currentRule] = 0.0;

                lp.row[currentRule+1] = -1.0;
                lp.row[currentWeight] = -f.values[i];
                if (addConstantBasis) lp.row[constBasisId] = -constBasisCoeff;
                lp.pushRow(LP::Constraint::Equal, 0.0);
                lp.row[currentRule+1] = 0.0;

                newFactor->getData().emplace_back(i, currentRule);
                currentRule += 2;
            }
            lp.row[currentWeight++] = 0.0;
        }
        lp.row[constBasisId] = 0.0;

        // Here signs are opposite to those of C since we need to find (Cw - b)
        // and (b - Cw)
        for (const auto & f : b.bases) {
            auto newFactor = graph.getFactor(f.tag);

            for (int i = 0; i < f.values.size(); ++i) {
                lp.row[currentRule] = 1.0;
                lp.pushRow(LP::Constraint::Equal, -f.values[i]);
                lp.row[currentRule] = 0.0;

                lp.row[currentRule+1] = 1.0;
                lp.pushRow(LP::Constraint::Equal, f.values[i]);
                lp.row[currentRule+1] = 0.0;

                newFactor->getData().emplace_back(i, currentRule);
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
        VE ve;
        Global global{lp, phiId, 0};

        ve(S, graph, global);

        // Set every variable we have added as unbounded, since they shouldn't
        // be limited to positive (which may be the default).
        for (int i = 0; i < lp.row.size(); ++i)
            lp.setUnbounded(i);

        // Finally, try to solve the LP and find out the coefficients to
        // approximate b using C!
        return lp.solve(phiId);
    }

    // Here's the implementation for the specifics of this Variable Elimination setup.

    void Global::initNewFactor() {
        // Each new "factor" here is basically a pair of constraints, so we
        // create two columns.
        newFactor = lp.row.size();

        lp.addColumn();
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

        // Now do the reverse for all opposite rules (same rules +1) since we
        // have an absolute value in the LP.
        for (int i = lp.row.size() - 2; i >= 0; --i) {
            if (lp.row[i] != 0.0) {
                lp.row[i+1] = lp.row[i];
                lp.row[i] = 0.0;
            }
        }
        lp.pushRow(LP::Constraint::LessEqual, 0.0);
    }

    void Global::makeResult(VE::FinalFactors && finalFactors) {
        // Once we have eliminated everything, we add the two final two phi
        // rules for all remaining factors.
        lp.row.setZero();
        lp.row[phiId] = -1.0;

        for (const auto ruleId : finalFactors)
            lp.row[ruleId] = 1.0;

        lp.pushRow(LP::Constraint::LessEqual, 0.0);

        // Now do the reverse for all opposite rules (same rules +1)
        for (const auto ruleId : finalFactors) {
            lp.row[ruleId] = 0.0;
            lp.row[ruleId+1] = 1.0;
        }

        lp.pushRow(LP::Constraint::LessEqual, 0.0);
    }
}
