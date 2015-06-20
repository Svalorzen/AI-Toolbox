#include <AIToolbox/POMDP/Algorithms/Utils/WitnessLP_lpsolve.hpp>

#include <lpsolve/lp_lib.h>

namespace AIToolbox {
    namespace POMDP {
        // Row is initialized to cols+1 since lp_solve reads element from 1 onwards
        WitnessLP_lpsolve::WitnessLP_lpsolve(size_t s) : S(s), cols_(s+2), lp(make_lp(0,cols_), delete_lp), row(new REAL[cols_+1]) {
            set_verbose(lp.get(), SEVERE /*or CRITICAL*/); // Make lp shut up. Could redirect stream to /dev/null if needed.
            set_simplextype(lp.get(), SIMPLEX_DUAL_DUAL);
            // set_BFP(lp.get(), "../../libbfp_etaPFI.so"); // Not included in Debian package, speeds around 3x, but also crashes

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
             * best[0][0] * b0 + best[0][1] * b1 + ... - K - delta <= 0
             * best[1][0] * b0 + best[1][1] * b1 + ... - K - delta <= 0
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
            {
                REAL one = 1.0;
                set_obj_fnex(lp.get(), 1, &one, &cols_);
                set_maxim(lp.get());
            }

            // CONSTRAINT: This is the simplex constraint (beliefs sum to 1)
            {
                // Note: lp_solve reads elements from 1 onwards, so we don't set row[0]
                for ( int i = 1; i < cols_-1; ++i )
                    row[i] = 1.0;
                row[cols_-1] = 0.0; // magic coefficient
                row[cols_]   = 0.0; // delta coefficient
                // The cols value doesn't really do anything here, the whole row is read
                add_constraint(lp.get(), row.get(), EQ, 1.0);
            }

            // IMPORTANT: K is unbounded, since the value function may be negative.
            set_unbounded(lp.get(), cols_-1);

            row[cols_-1] = -1.0;
            row[cols_]   = +0.0;
        }

        void WitnessLP_lpsolve::addOptimalRow(const MDP::Values & v) {
            // Temporarily set the delta constraint
            row[cols_] = +1.0;
            pushRow(v, LE);
            row[cols_] = 0.0;
        }

        std::tuple<bool, POMDP::Belief> WitnessLP_lpsolve::findWitness(const MDP::Values & v) {
            // Add witness constraint
            pushRow(v, EQ);

            // lp_solve uses the result of the previous runs to bootstrap
            // the new solution. Sometimes this breaks down for some reason,
            // so we just avoid it - it does not really even give a performance
            // boost..
            default_basis(lp.get());

            // print_lp(lp.get());
            auto result = ::solve(lp.get());

            // Note: do not popRow here, or the pointer to the
            // solution will be lost!
            REAL * vp;
            get_ptr_variables(lp.get(), &vp);
            REAL value = get_objective(lp.get());

            // We have found a witness point if we have found a belief for which the value
            // of the supplied ValueFunction is greater than ALL others. Thus we just need
            // to verify that the variable we have minimized is actually less than 0.
            bool isSolved = !( result > 1 || value <= 0.0 );

            POMDP::Belief solution;

            if ( isSolved )
                solution = Eigen::Map<POMDP::Belief>(vp, S);

            popRow();
            return std::make_pair(isSolved, solution);
        }

        void WitnessLP_lpsolve::reset() {
            resize_lp(lp.get(), 1, cols_);
        }

        void WitnessLP_lpsolve::allocate(size_t rows) {
            resize_lp(lp.get(), rows+1, cols_);
        }

        void WitnessLP_lpsolve::pushRow(const MDP::Values & v, int constrType) {
            for ( size_t s = 0; s < S; ++s )
                row[s+1] = v[s];

            add_constraint(lp.get(), row.get(), constrType, 0.0);
        }

        void WitnessLP_lpsolve::popRow() {
            del_constraint(lp.get(), get_Nrows(lp.get()));
        }
    }
}
