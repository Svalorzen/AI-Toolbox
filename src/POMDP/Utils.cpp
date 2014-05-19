#include <AIToolbox/POMDP/Utils.hpp>

#include <utility>

#include <lpsolve/lp_lib.h>

#include <AIToolbox/POMDP/Types.hpp>

namespace AIToolbox {
    namespace POMDP {
        VList crossSum(const VList & a, const VList & b) {
            VList c;

            for ( auto & va : a )
                for ( auto & vb : b ) {
                    MDP::ValueFunction v(va.size());
                    for ( size_t i = 0; i < va.size(); ++i )
                        v[i] = va[i] + vb[i];
                    c.push_back(v);
                }

             return c;
        }

        // The idea is that the input thing already has all the best vectors,
        // thus we only need to find them and discard the others.
        void prune(VList * pw) {
            auto & w = *pw;

            if ( w.size() < 2 ) return;
            size_t S = w[0].size();

            // Remove easy ValueFunctions to avoid doing more work later.
            dominationPrune(&w);

            // Initialize the new best list with some easy finds, and remove them from
            // the old list.
            auto bestBound = findBestAtSimplexCorners(S, std::begin(w), std::end(w));
            VList best(std::make_move_iterator(bestBound), std::make_move_iterator(std::end(w)));
            w.erase(bestBound, std::end(w));

            // For each of the remaining points now we try to find a witness point with respect
            // to the best ones. If there is, there is something we need to extract to best.
            VList::iterator begin = std::begin(w), end = std::end(w);
            while ( begin < end ) {
                auto result = findWitnessPoint(*begin, best);
                // If we get a belief point, we search for the actual vector that provides
                // the best value on the belief point, we move it into the best vector.
                if ( std::get<0>(result) ) {
                    auto bestMatch = findBestVector( std::get<1>(result), begin, end );
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
        void dominationPrune(VList * pw) {
            auto & w = *pw;
            if ( w.size() < 2 ) return;

            // We use this comparison operator to filter all dominated vectors.
            // We define a vector to be dominated by an equal vector, so that
            // we can remove duplicates in a single swoop. However, we avoid
            // removing everything by returning false for comparison of the vector with itself.
            struct {
                const MDP::ValueFunction * rhs;
                bool operator()(const MDP::ValueFunction & lhs) {
                    if ( &lhs == rhs ) return false;
                    for ( size_t i = 0; i < lhs.size(); ++i )
                        if ( (*rhs)[i] > lhs[i] ) return false;
                    return true;
                }
            } dominates;

            // For each vector, if we find a vector that dominates it, then we remove it.
            // Otherwise we continue, comparing every vector with every other non-dominated
            // vector.
            VList::iterator begin = std::begin(w), end = std::end(w), iter = begin, helper;
            while ( iter < end ) {
                dominates.rhs = &(*iter);
                helper = std::find_if(begin, end, dominates);
                if ( helper != end )
                    std::swap( *iter, *(--end) );
                else
                    ++iter;
            }

            // Cleanup of dominated vectors.
            w.erase(end, std::end(w));
        }

        double dotProd(const MDP::ValueFunction & a, const MDP::ValueFunction & b) {
            double result = 0.0;

            for ( size_t i = 0; i < a.size(); ++i )
                result += a[i] * b[i];

            return result;
        }

        VList::iterator findBestAtSimplexCorners(size_t S, VList::iterator begin, VList::iterator end) {
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

        VList::iterator findBestVector(const Belief & belief, VList::iterator begin, VList::iterator end) {
            auto bestMatch = begin;
            double bestValue = dotProd(belief, *bestMatch);

            while ( (++begin) < end ) {
                double currValue = dotProd(belief, *begin);
                if ( currValue > bestValue || ( currValue == bestValue && ( *begin > *bestMatch ) ) ) {
                    bestMatch = begin;
                    bestValue = currValue;
                }
            }

            return bestMatch;
        }

        std::pair<bool, Belief> findWitnessPoint(const MDP::ValueFunction & v, const VList & best) {
            size_t S = v.size();
            // If there's nothing to compare to, any belief point is a witness.
            if ( best.size() == 0 ) return std::make_pair(true, Belief(S, 1.0/S));

            // We have a column per state + one for the maximizing parameter.
            int states = static_cast<int>(S);
            int cols = states + 1;

            // Setup linear programming problem
            auto lp = make_lp(0,cols);
            set_verbose(lp, SEVERE /*or CRITICAL*/); // Make lp shut up. Could redirect stream to /dev/null if needed.

            // Actually allocate memory, but we get to use the add_constraintex functs.
            resize_lp(lp, best.size() + states + 1, cols);

            // (v[0] - best[0][0]) * b0 + (v[1] - best[0][1]) * b1 + ... - delta >= 0
            // (v[0] - best[1][0]) * b0 + (v[1] - best[1][1]) * b1 + ... - delta >= 0
            // ...
            // b0 >= 0
            // b1 >= 0
            // ...
            // b0 + b1 + ... + bn = 1.0

            // Goal: maximize delta.
            {
                REAL one = 1.0;
                set_obj_fnex(lp, 1, &one, &cols);
                set_maxim(lp);
            }

            // Start adding all the rows
            set_add_rowmode(lp, TRUE);

            // CONSTRAINTS: Here are the individual belief constraints (>=0)
            {
                REAL beliefRow = 1.0;

                // Each belief element needs to be greater or equal to zero (columns are 1-numbered)
                for ( int i = 1; i < states + 1; ++i )
                    add_constraintex(lp, 1, &beliefRow, &i, GE, 0.0);
            }

            // We will use this also to extract the solution.
            auto row   = new REAL[cols+1]; // lp constraints start from element 1 so we need one more.
            {
                // CONSTRAINTS: This is the simplex constraint (beliefs sum to 1)
                row[cols] = 0.0; // delta coefficient
                for ( int i = 1; i < cols; ++i )
                    row[i] = 1.0;
                // The cols value doesn't really do anything here, the whole row is read
                add_constraintex(lp, cols, row, NULL, EQ, 1.0);


                // CONSTRAINTS: This are the value functions constraints.
                row[cols] = -1.0; // delta coefficient
                for ( auto & b : best ) {
                    for ( int i = 0; i < states; ++i )
                        row[i+1] = v[i] - b[i];

                    add_constraintex(lp, cols, row, NULL, GE, 0.0);
                }

            }
            set_add_rowmode(lp, FALSE);

            // print_lp(lp);
            auto result = solve(lp);

            get_variables(lp, row);
            REAL value = get_objective(lp);

            delete_lp(lp);

            // We have found a witness point if we have found a belief for which the value
            // of the supplied ValueFunction is greater than ALL others. Thus we just need
            // to verify that the variable we have minimized is actually less than 0.
            if ( result || value <= 0.0 || row[cols-1] <= 0.0 ) {
                delete[] row;
                return std::make_pair(false, Belief());
            }

            Belief solution(row, row + states);
            delete[] row;
            return std::make_pair(true, solution);
        }
    }
}
