#include <AIToolbox/POMDP/Algorithms/Utils/WitnessLP_clp.hpp>

#include <iterator>
#include <iostream>

namespace AIToolbox {
    namespace POMDP {
        WitnessLP_clp::WitnessLP_clp(size_t s) : S(s), cols_(s+2), indeces(new int[cols_]), row(new double[cols_]) {
            // Verbosity
            lp.setLogLevel(0);
            // lp.setPersistenceFlag(1);
            // Initialize indeces
            std::iota(indeces.get(), indeces.get() + cols_, 0);
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
             * 1) The simplex constraint is the only constraint that never changes,
             * so we are going to set it up now.
             *
             * 2) The other constraints can be rewritten as follows:
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
            lp.resize( 0, cols_ );

            // Setup beliefs
            for ( auto i = 0; i < cols_-2; ++i ) {
                lp.setObjCoeff(i, 0.0);
                // Force belief elements between 0 and 1
                lp.setColumnBounds(i, 0.0, 1.0);
            }

            // Setup K (unbounded)
            lp.setObjCoeff(cols_-2, 0.0);
            lp.setColumnBounds(cols_-2, -COIN_DBL_MAX, +COIN_DBL_MAX);

            // Setup delta
            lp.setObjCoeff(cols_-1, -1.0); // maximize delta via minimization
            lp.setColumnBounds(cols_-1, -COIN_DBL_MAX, +COIN_DBL_MAX);

            // CONSTRAINT: This is the simplex constraint (beliefs sum to 1)
            {
                // Note: lp_solve reads elements from 1 onwards, so we don't set row[0]
                for ( int i = 0; i < cols_-2; ++i )
                    row[i] = 1.0;
                row[cols_-2] = 0.0; // magic coefficient
                row[cols_-1] = 0.0; // delta coefficient
                // The cols_ value doesn't really do anything here, the whole row is read
                lp.addRow(cols_, indeces.get(), row.get(), 1.0, 1.0);
            }

            row[cols_-2] = -1.0;
            row[cols_-1] = +0.0;
        }

        void WitnessLP_clp::addOptimalRow(const std::vector<double> & v) {
            // Temporarily set the delta constraint
            row[cols_-1] = +1.0;
            pushRow(v, -COIN_DBL_MAX, 0.0); // Less equal than zero
            row[cols_-1] = 0.0;
        }

        std::tuple<bool, POMDP::Belief> WitnessLP_clp::findWitness(const std::vector<double> & v) {
            // Add witness constraint
            pushRow(v, 0.0, 0.0); // Equal to zero

            lp.dual();
            lp.primal();

            // TODO: There's a function that gets a pointer to the solution
            // stored within the LP, maybe use that? (get_ptr_primal_solution)
            auto results = lp.primalColumnSolution();
            auto value =   lp.objectiveValue();

            bool isOptimal = lp.isProvenOptimal();

            // Remove test row
            popRow();

            // We have found a witness point if we have found a belief for which the value
            // of the supplied ValueFunction is greater than ALL others. Thus we just need
            // to verify that the variable we have minimized is actually less than 0.
            if ( ! isOptimal || value >= 0.0 ) {
                return std::make_pair(false, POMDP::Belief());
            }

            // For some reason when lp_solve returns the variables it puts them from
            // 0 to S-1, so here we read accordingly.
            POMDP::Belief solution(results, results + S);
            return std::make_pair(true, solution);
        }

        void WitnessLP_clp::reset() {
            lp.resize( 1, cols_ );
        }

        void WitnessLP_clp::allocate(size_t) {}

        void WitnessLP_clp::pushRow(const std::vector<double> & v, double min, double max) {
            for ( size_t s = 0; s < S; ++s )
                row[s] = v[s];

            lp.addRow(cols_, indeces.get(), row.get(), min, max);
        }

        void WitnessLP_clp::popRow() {
            int numRows = lp.getNumRows() - 1;
            lp.deleteRows(1, &numRows);
        }
    }
}
