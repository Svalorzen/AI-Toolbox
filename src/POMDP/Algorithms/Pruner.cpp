#include <AIToolbox/POMDP/Algorithms/Pruner.hpp>

#include <lpsolve/lp_lib.h>

namespace AIToolbox {
    namespace POMDP {
        // THIS IS A TEMPORARY FUNCTION UNTIL WE SWITCH TO UBLAS
        double dotProd(size_t S, const MDP::ValueFunction & a, const MDP::ValueFunction & b) {
            double result = 0.0;

            for ( size_t i = 0; i < S; ++i )
                result += a[i] * b[i];

            return result;
        }

        // Row is initialized to cols+1 since lp_solve reads element from 1 onwards
        Pruner::Pruner(size_t s) : S(s), cols(s+1), lp(make_lp(0, cols), delete_lp), row(new REAL[cols+1]) {
            set_verbose(lp.get(), SEVERE /*or CRITICAL*/); // Make lp shut up. Could redirect stream to /dev/null if needed.

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
             * So here we are going to setup 'delta' and all the belief constraints,
             * while we will modify the others at each 'findWitnessPoint' call.
             *
             * In particular by default no variable will assume negative values, so
             * the only constraint that we need to put is the simplex one.
             *
             */

            // Make space for the simplex constraint.
            resize_lp(lp.get(), 1, cols);

            // Goal: maximize delta.
            {
                REAL one = 1.0;
                set_obj_fnex(lp.get(), 1, &one, &cols);
                set_maxim(lp.get());
            }

            // Start adding all the rows
            set_add_rowmode(lp.get(), TRUE);

            // CONSTRAINTS: This is the simplex constraint (beliefs sum to 1)
            {
                // Note: lp_solve reads elements from 1 onwards, so we don't set row[0]
                row[cols] = 0.0; // delta coefficient
                for ( int i = 1; i < cols; ++i )
                    row[i] = 1.0;
                // The cols value doesn't really do anything here, the whole row is read
                add_constraintex(lp.get(), cols, row.get(), NULL, EQ, 1.0);
            }
        }

        // The idea is that the input thing already has all the best vectors,
        // thus we only need to find them and discard the others.
        void Pruner::operator()(VList * pw) {
            auto & w = *pw;

            // Remove easy ValueFunctions to avoid doing more work later.
            dominationPrune(&w);

            if ( w.size() < 2 ) return;

            // Initialize the new best list with some easy finds, and remove them from
            // the old list.
            auto bestBound = findBestAtSimplexCorners(std::begin(w), std::end(w));
            VList best(std::make_move_iterator(bestBound), std::make_move_iterator(std::end(w)));

            // For each of the remaining points now we try to find a witness point with respect
            // to the best ones. If there is, there is something we need to extract to best.
            VList::iterator begin = std::begin(w), end = bestBound;
            while ( begin < end ) {
                auto result = findWitnessPoint( begin->second, best );
                // If we get a belief point, we search for the actual vector that provides
                // the best value on the belief point, we move it into the best vector.
                if ( std::get<0>(result) ) {
                    auto bestMatch = findBestVector(std::get<1>(result), begin, end);
                    std::swap( *bestMatch, *(--end) );
                    best.emplace_back(std::move(*end)); // We don't care about what we leave there..
                }
                // We only advance if we did not find anything. Otherwise, we may have found a
                // witness point for the current value, but since we are not guaranteed to have
                // put into best that value, it may still keep witness to other belief points!
                else {
                    ++begin;
                }
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
                const MDP::ValueFunction * rhs;
                size_t S;
                bool operator()(const std::pair<size_t, MDP::ValueFunction> & lhs) {
                    if ( &(lhs.second) == rhs ) return false;
                    for ( size_t i = 0; i < S; ++i )
                        if ( (*rhs)[i] > lhs.second[i] ) return false;
                    return true;
                }
            } dominates;

            dominates.S = S;

            // For each vector, if we find a vector that dominates it, then we remove it.
            // Otherwise we continue, comparing every vector with every other non-dominated
            // vector.
            VList::iterator begin = std::begin(w), end = std::end(w), iter = begin, helper;
            while ( iter < end ) {
                dominates.rhs = &(iter->second);
                helper = std::find_if(begin, end, dominates);
                if ( helper != end )
                    std::swap( *iter, *(--end) );
                else
                    ++iter;
            }

            // Cleanup of dominated vectors.
            w.erase(end, std::end(w));
        }

        VList::iterator Pruner::findBestAtSimplexCorners(VList::iterator begin, VList::iterator end) {
            if ( begin == end ) return begin;
            // Setup the corners. The additional space is to avoid
            // writing in unallocated space in the last loop iteration.
            Belief corner(S+1, 0.0);
            corner[0] = 1.0;

            auto bestBound = end;
            // For each corner
            for ( size_t s = 1; s <= S; ++s ) {
                // We are going to move the best ones at the end, if they are not
                // already there. This will allow us to move everything easily later.
                auto bestMatch = findBestVector(corner, begin, end);
                if ( bestMatch < bestBound )
                    std::swap(*bestMatch, *(--bestBound));
                std::swap(corner[s-1], corner[s]); // change corner
            }

            return bestBound;
        }

        VList::iterator Pruner::findBestVector(const Belief & belief, VList::iterator begin, VList::iterator end) {
            auto bestMatch = begin;
            double bestValue = dotProd(S, belief, bestMatch->second);

            while ( (++begin) < end ) {
                double currValue = dotProd(S, belief, begin->second);
                if ( currValue > bestValue || ( currValue == bestValue && ( *begin > *bestMatch ) ) ) {
                    bestMatch = begin;
                    bestValue = currValue;
                }
            }

            return bestMatch;
        }

        std::pair<bool, Belief> Pruner::findWitnessPoint(const MDP::ValueFunction & v, const VList & best) {
            // If there's nothing to compare to, any belief point is a witness.
            if ( best.size() == 0 ) return std::make_pair(true, Belief(S, 1.0/S));

            int newRows = best.size() + 1;
            // Remove old constraints and setup lp again.
            resize_lp(lp.get(), 1, cols);
            resize_lp(lp.get(), newRows, cols);

            /*
             * Here we finish up the constraints left.
             *
             * (v[0] - best[0][0]) * b0 + (v[1] - best[0][1]) * b1 + ... - delta >= 0
             * (v[0] - best[1][0]) * b0 + (v[1] - best[1][1]) * b1 + ... - delta >= 0
             * ...
             *
             */

            // Start adding all the rows
            // set_add_rowmode(lp.get(), TRUE);

            // CONSTRAINTS: This are the value functions constraints.
            {
                // delta coefficient
                row[cols] = -1.0;
                // This keeps track of which vector we are adding
                size_t vectorCounter = 0;

                // For all others, we instead use add_constraintex
                for ( int i = 2; i <= newRows; ++i ) {
                    for ( int col = 1; col < cols; ++col )
                        row[col] = v[col-1] - best[vectorCounter].second[col-1];

                    add_constraintex(lp.get(), cols, row.get(), NULL, GE, 0.0);
                    ++vectorCounter;
                }
            }
            set_add_rowmode(lp.get(), FALSE);

            // print_lp(lp.get());
            auto result = solve(lp.get());

            get_variables(lp.get(), row.get());
            REAL value = get_objective(lp.get());

            // We have found a witness point if we have found a belief for which the value
            // of the supplied ValueFunction is greater than ALL others. Thus we just need
            // to verify that the variable we have minimized is actually less than 0.
            if ( result || value <= 0.0 || row[cols-1] <= 0.0 ) {
                return std::make_pair(false, Belief());
            }

            // For some reason when lp_solve returns the variables it puts them from
            // 0 to S-1, so here we read accordingly.
            Belief solution(row.get(), row.get() + cols - 1);
            return std::make_pair(true, solution);
        }
    }
}
