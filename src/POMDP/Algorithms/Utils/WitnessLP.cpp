#include <AIToolbox/POMDP/Algorithms/Utils/WitnessLP.hpp>

#include <lpsolve/lp_lib.h>

namespace AIToolbox::POMDP {
    // Row is initialized to cols+1 since lp_solve reads element from 1 onwards
    WitnessLP::WitnessLP(const size_t s) : S(s), lp_(s+2)
    {
        /*
         * Here we setup the part of the lp that never changes (at least with this number of states)
         * Our constraints are of the form
         *
         * b0 >= 0
         * b1 >= 0
         * ...
         * b0 + b1 + ... + bn = 1.0
         * (v[0] - best[0][0]) * b0 + (v[1] - best[0][1]) * b1 + ... - delta >= 0
         * (v[0] - best[1][0]) * b0 + (v[1] - best[1][1]) * b1 + ... - delta >= 0
         * ...
         *
         * In particular we know that:
         *
         * 1) By default no variable will assume negative values, so
         * the only constraint that we need to put is the simplex one,
         * as lp_solve automatically sets all variables to be positive.
         *
         * 2) The simplex constraint is the only constraint that never changes,
         * so we are going to set it up now.
         *
         * 3) The other constraints can be rewritten as follows:
         *
         *       v[0] * b0 +       v[1] * b1 + ... - K          = 0
         * best[0][0] * b0 + best[0][1] * b1 + ... - K + delta <= 0
         * best[1][0] * b0 + best[1][1] * b1 + ... - K + delta <= 0
         * ...
         *
         * Where basically with the first constraint we are setting K
         * to the value of v in the final belief, and we are forcing
         * all other values to be less than that by forcing:
         *
         * delta > 0
         *
         * The client can then add these and remove them at their discretion.
         */

        // Goal: maximize delta.
        lp_.setObjective(S+1, true);

        // CONSTRAINT: This is the simplex constraint (beliefs sum to 1)
        {
            // Note: lp_solve reads elements from 1 onwards, so we don't set row[0]
            for ( size_t i = 0; i < S; ++i )
                lp_.row[i] = 1.0;
            lp_.row[S]     = 0.0; // magic coefficient
            lp_.row[S + 1] = 0.0; // delta coefficient
            // The cols value doesn't really do anything here, the whole row is read
            lp_.pushRow(LP::Constraint::Equal, 1.0);
        }

        // IMPORTANT: K is unbounded, since the value function may be negative.
        lp_.setUnbounded(S);

        lp_.row[S]     = -1.0;
        lp_.row[S + 1] = +0.0;
    }

    void WitnessLP::addOptimalRow(const MDP::Values & v) {
        for ( size_t i = 0; i < S; ++i )
            lp_.row[i] = v[i];
        // Temporarily set the delta constraint
        lp_.row[S+1] = +1.0;
        lp_.pushRow(LP::Constraint::LessEqual, 0.0);

        lp_.row[S+1] = 0.0;
    }

    std::optional<POMDP::Belief> WitnessLP::findWitness(const MDP::Values & v) {
        // Add witness constraint
        for ( size_t i = 0; i < S; ++i )
            lp_.row[i] = v[i];
        lp_.pushRow(LP::Constraint::Equal, 0.0);

        auto solution = lp_.solve(S);

        // Remove it
        lp_.popRow();

        return solution;
    }

    void WitnessLP::reset() {
        lp_.resize(1);
    }

    void WitnessLP::allocate(const size_t rows) {
        lp_.resize(rows+1);
    }
}
