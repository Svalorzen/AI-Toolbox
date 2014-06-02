#include <AIToolbox/POMDP/Algorithms/Pruner.hpp>

#include <AIToolbox/POMDP/Utils.hpp>

#include <lpsolve/lp_lib.h>

namespace AIToolbox {
    namespace POMDP {
        // Row is initialized to cols+1 since lp_solve reads element from 1 onwards
        Pruner::Pruner(size_t s) : S(s), cols(s+2), lp(make_lp(0,cols), delete_lp), row(new REAL[cols+2]) {
            set_verbose(lp.get(), SEVERE /*or CRITICAL*/); // Make lp shut up. Could redirect stream to /dev/null if needed.
            // set_BFP(lp.get(), "../libbfp_etaPFI.so"); Not included in Debian package, speeds around 3x

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
             * the only constraint that we need to put is the simplex one.
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
             * all other values to be less than that.
             *
             * It is important to notice that those constraints never change,
             * they only increase one at a time (aside from the 'v' constraint).
             * Thus what we are going to do is to push each 'best' constraint
             * into the lp, and then push/pop the 'v' constraint every time
             * we need to try out a new one.
             *
             * That we do in the findWitnessPoint function.
             *
             */

            // Goal: maximize delta.
            {
                REAL one = 1.0;
                set_obj_fnex(lp.get(), 1, &one, &cols);
                set_maxim(lp.get());
            }

            // CONSTRAINTS: This is the simplex constraint (beliefs sum to 1)
            {
                // Note: lp_solve reads elements from 1 onwards, so we don't set row[0]
                for ( int i = 1; i < cols-1; ++i )
                    row[i] = 1.0;
                row[cols-1] = 0.0; // magic coefficient
                row[cols]   = 0.0; // delta coefficient
                // The cols value doesn't really do anything here, the whole row is read
                add_constraintex(lp.get(), cols, row.get(), NULL, EQ, 1.0);
            }

            // IMPORTANT: K is unbounded, since the value function may be negative.
            set_unbounded(lp.get(), cols-1);

            row[cols-1] = -1.0;
            row[cols]   = +1.0;
        }

        void Pruner::setLP(size_t rows) {
            // Here we simply remove all constraints that are not the simplex
            // one in order to reset the lp without reallocations.
            resize_lp(lp.get(), 1, cols);
            resize_lp(lp.get(), rows, cols);
        }

        void Pruner::addRow(const MDP::Values & v, int constrType) {
            for ( size_t s = 0; s < S; ++s )
                row[s+1] = v[s];

            add_constraintex(lp.get(), cols, row.get(), NULL, constrType, 0.0);
        }

        void Pruner::popRow() {
            del_constraint(lp.get(), get_Nrows(lp.get()));
        }

        // The idea is that the input thing already has all the best vectors,
        // thus we only need to find them and discard the others.
        void Pruner::operator()(VList * pw) {
            auto & w = *pw;

            // Remove easy ValueFunctions to avoid doing more work later.
            dominationPrune(&w);

            size_t size = w.size();
            if ( size < 2 ) return;
            // We setup the lp preparing for a max of size rows.
            setLP(size);

            // Initialize the new best list with some easy finds, and remove them from
            // the old list.
            VList::iterator begin = std::begin(w), end = std::end(w), bound = end;

            bound = extractBestAtSimplexCorners(begin, bound, end);
            // Here we could do some random belief lookups..

            // Initialize best list with what we have found so far.
            VList best(std::make_move_iterator(bound), std::make_move_iterator(end));

            // Setup initial LP rows.
            for ( auto & bv : best )
                addRow(std::get<VALUES>(bv), LE);

            // For each of the remaining points now we try to find a witness point with respect
            // to the best ones. If there is, there is something we need to extract to best.
            while ( begin < bound ) {
                auto result = findWitnessPoint( std::get<VALUES>(*begin), best );
                // If we get a belief point, we search for the actual vector that provides
                // the best value on the belief point, we move it into the best vector.
                if ( std::get<0>(result) ) {
                    bound = extractBestAtBelief(std::get<1>(result), begin, bound, bound);  // Moves the best at the "end"
                    best.emplace_back(std::move(*bound));                                   // We don't care about what we leave here..
                    addRow(std::get<VALUES>(best.back()), LE);                              // Add the newly found vector to our lp.
                }
                // We only advance if we did not find anything. Otherwise, we may have found a
                // witness point for the current value, but since we are not guaranteed to have
                // put into best that value, it may still keep witness to other belief points!
                else
                    ++begin;
            }

            // Finally, we discard all bad vectors (and remains of moved ones) and
            // we return just the best list.
            std::swap(w, best);
        }

        // We need to get the list otherwise we cannot erase it.
        void Pruner::dominationPrune(VList * pw) {
            auto & w = *pw;
            if ( w.size() < 2 ) return;

            // We use this comparison operator to filter all dominated vectors.
            // We define a vector to be dominated by an equal vector, so that
            // we can remove duplicates in a single swoop. However, we avoid
            // removing everything by returning false for comparison of the vector with itself.
            struct {
                const MDP::Values * rhs;
                size_t S;
                bool operator()(const VEntry & lhs) {
                    auto & lhsV = std::get<VALUES>(lhs);
                    if ( &(lhsV) == rhs ) return false;
                    for ( size_t i = 0; i < S; ++i )
                        if ( (*rhs)[i] > lhsV[i] ) return false;
                    return true;
                }
            } dominates;

            dominates.S = S;

            // For each vector, if we find a vector that dominates it, then we remove it.
            // Otherwise we continue, comparing every vector with every other non-dominated
            // vector.
            VList::iterator begin = std::begin(w), end = std::end(w), iter = begin, helper;
            while ( iter < end ) {
                dominates.rhs = &(std::get<VALUES>(*iter));
                helper = std::find_if(begin, end, dominates);
                if ( helper != end )
                    std::swap( *iter, *(--end) );
                else
                    ++iter;
            }

            // Cleanup of dominated vectors.
            w.erase(end, std::end(w));
        }

        VList::iterator Pruner::extractBestAtSimplexCorners(VList::iterator begin, VList::iterator bound, VList::iterator end) {
            if ( begin == bound ) return bound;
            // Setup the corners. The additional space is to avoid
            // writing in unallocated space in the last loop iteration.
            Belief corner(S+1, 0.0);
            corner[0] = 1.0;

            // For each corner
            for ( size_t s = 1; s <= S; ++s ) {
                // FIXME: This incrementally adds corners. Since a corner comparison
                // simply checks a single ValueFunction value, this can be implemented way faster.
                bound = extractBestAtBelief(corner, begin, bound, end);
                std::swap(corner[s-1], corner[s]); // change corner
            }

            return bound;
        }

        VList::iterator Pruner::extractBestAtBelief(const Belief & belief, VList::iterator begin, VList::iterator bound, VList::iterator end) {
            auto bestMatch = findBestAtBelief(S, belief, begin, end);

            if ( bestMatch < bound )
                std::swap(*bestMatch, *(--bound));

            return bound;
        }

        std::pair<bool, Belief> Pruner::findWitnessPoint(const MDP::Values & v, const VList & best) {
            // If there's nothing to compare to, any belief point is a witness.
            if ( best.size() == 0 ) return std::make_pair(true, Belief(S, 1.0/S));

            // We push the v constraint on to the "stack"
            row[cols] = 0.0;
            addRow(v, EQ);

            // print_lp(lp.get());
            auto result = solve(lp.get());

            // TODO: There's a function that gets a pointer to the solution
            // stored within the LP, maybe use that? (get_ptr_primal_solution)
            get_variables(lp.get(), row.get());
            REAL value = get_objective(lp.get());

            // And we pop it at the end.
            row[cols-1] = -1.0;
            row[cols]   = +1.0;
            popRow();

            // We have found a witness point if we have found a belief for which the value
            // of the supplied ValueFunction is greater than ALL others. Thus we just need
            // to verify that the variable we have minimized is actually less than 0.
            if ( result || value <= 0.0 ) {
                return std::make_pair(false, Belief());
            }

            // For some reason when lp_solve returns the variables it puts them from
            // 0 to S-1, so here we read accordingly.
            Belief solution(row.get(), row.get() + cols - 1);
            return std::make_pair(true, solution);
        }
    }
}
