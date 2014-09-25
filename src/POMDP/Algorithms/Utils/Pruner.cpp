#include <AIToolbox/POMDP/Algorithms/Utils/Pruner.hpp>

#include <AIToolbox/POMDP/Utils.hpp>

#include <lpsolve/lp_lib.h>

namespace AIToolbox {
    namespace POMDP {
        Pruner::Pruner(size_t s) : S(s), lp(s) {}

        // The idea is that the input thing already has all the best vectors,
        // thus we only need to find them and discard the others.
        void Pruner::operator()(VList * pw) {
            auto & w = *pw;

            // Remove easy ValueFunctions to avoid doing more work later.
            dominationPrune(S, &w);

            size_t size = w.size();
            if ( size < 2 ) return;
            // We setup the lp preparing for a max of size rows.
            lp.setRowNr(size);

            // Initialize the new best list with some easy finds, and remove them from
            // the old list.
            VList::iterator begin = std::begin(w), end = std::end(w), bound = end;

            bound = extractBestAtSimplexCorners(begin, bound, end);
            // Here we could do some random belief lookups..

            // Initialize best list with what we have found so far.
            VList best(std::make_move_iterator(bound), std::make_move_iterator(end));

            // Setup initial LP rows.
            for ( auto & bv : best )
                lp.addRow(std::get<VALUES>(bv), LE);

            // For each of the remaining points now we try to find a witness
            // point with respect to the best ones. If there is, there is
            // something we need to extract to best.
            //
            // What we are going to do is to push each 'best' constraint into
            // the lp, and then push/pop the 'v' constraint every time we need
            // to try out a new one.
            //
            // That we do in the findWitnessPoint function.
            while ( begin < bound ) {
                auto result = findWitnessPoint( std::get<VALUES>(*begin), best );
                // If we get a belief point, we search for the actual vector that provides
                // the best value on the belief point, we move it into the best vector.
                if ( std::get<0>(result) ) {
                    bound = extractBestAtBelief(S, std::get<1>(result), begin, bound, bound);  // Moves the best at the "end"
                    best.emplace_back(std::move(*bound));                                      // We don't care about what we leave here..
                    lp.addRow(std::get<VALUES>(best.back()), LE);                              // Add the newly found vector to our lp.
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
        void Pruner::dominationPrune(size_t S, VList * pw) {
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
                bound = extractBestAtBelief(S, corner, begin, bound, end);
                std::swap(corner[s-1], corner[s]); // change corner
            }

            return bound;
        }

        std::tuple<bool, Belief> Pruner::findWitnessPoint(const MDP::Values & v, const VList & best) {
            // If there's nothing to compare to, any belief point is a witness.
            if ( best.size() == 0 ) return std::make_pair(true, Belief(S, 1.0/S));

            // We push the v constraint on to the "stack"
            lp.setDeltaCoefficient(0.0);
            lp.addRow(v, EQ);

            auto result = lp.solve();

            // Remove tested constraint and set delta back
            lp.setDeltaCoefficient(+1.0);
            lp.popRow();

            return result;
        }
    }
}
