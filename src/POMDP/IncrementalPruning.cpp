#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>

#include <list>
#include <utility>

#include <iostream>

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

            dominationPrune(&w);
            // Initialize best with the easy ones.
            VList best = extractBestAtCorners(&w);

            for ( size_t i = 0; i < w.size(); ++i ) {
                auto result = findWitnessPoint(w[i], best);
                if ( std::get<0>(result) ) {
                    size_t bestMatch = findBestVector( std::get<1>(result), w, i, w.size() );
                    best.emplace_back(std::move(w[bestMatch])); // We don't care about what we leave there..
                }
            }
            // Save best ones in return VList
            std::swap(w, best);
        }

        void dominationPrune(VList * pw) {
            auto & w = *pw;

            if ( w.size() < 2 ) return;

            struct {
                MDP::ValueFunction * comp = nullptr;
                bool operator()(const MDP::ValueFunction & v) {
                    for ( size_t i = 0; i < v.size(); ++i )
                        if ( v[i] > (*comp)[i] ) return false;
                    return true;
                }
            } isDominated;

            auto iter = w.begin();
            do {
                isDominated.comp = &(*iter);
                std::remove_if(++(w.begin()), w.end(), isDominated);
                ++iter;
            }
            while ( iter != w.end() );
        }

        double dotProd(const MDP::ValueFunction & a, const MDP::ValueFunction & b) {
            double result = 0.0;

            for ( size_t i = 0; i < a.size(); ++i )
                result += a[i] * b[i];

            return result;
        }

        VList extractBestAtCorners(VList * pw) {
            auto & w = *pw;
            size_t S = w.front().size();

            // Setup the corners
            Belief corner(S, 0.0);
            corner[0] = 1.0;

            // We are going to keep track of all the elements that we need to remove
            // and remove them all at once.
            struct {
                std::vector<bool> marks;
                bool operator()(const MDP::ValueFunction &) const {
                    static int counter = -1;
                    ++counter;
                    return marks[counter];
                }
            } remover;

            remover.marks.reserve(w.size());

            VList best;
            size_t bestSize = 0;
            // For each corner
            for ( size_t s = 1; s <= S; ++s ) {
                size_t bestMatch = findBestVector(corner, w, 0, w.size());
                if ( !remover.marks[bestMatch] ) {
                    remover.marks[bestMatch] = true;
                    ++bestSize;
                }
                std::swap(corner[s-1], corner[s]); // Move corner
            }

            best.reserve(bestSize);
            // Move elements away
            std::copy_if(std::make_move_iterator(std::begin(w)),
                         std::make_move_iterator(std::end(w)), std::back_inserter(best), remover);
            // Remove stuff left
            w.erase(std::remove_if(std::begin(w), std::end(w), remover), std::end(w));

            return best;
        }

        size_t findBestVector(const Belief & belief, const VList & w, size_t start, size_t end) {
            size_t bestMatch = start;
            double bestValue = dotProd(belief, w[bestMatch]);

            for ( size_t i = start + 1; i < end; ++i ) {
                double currValue = dotProd(belief, w[i]);
                if ( currValue > bestValue || ( currValue == bestValue && w[i] > w[bestMatch] ) ) {
                    bestMatch = i;
                    bestValue = currValue;
                }
            }

            return bestMatch;
        }

        std::pair<bool, Belief> findWitnessPoint(const MDP::ValueFunction & v, const VList & best) {
            size_t S = v.size();
            // If there's nothing to compare to, any belief point is a witness.
            if ( best.size() == 0 ) return std::make_pair(true, Belief(S, 1.0/S));

            // Setup linear programming problem

            // We have a column per state + one for the maximizing parameter.
            int states = static_cast<int>(S);
            int cols = states + 1;

            auto lp = make_lp(best.size() + states + 1,cols);

            // (v[0] - best[0][0]) * b0 + (v[1] - best[0][1]) * b1 + ... - delta >= 0
            // (v[0] - best[1][0]) * b0 + (v[1] - best[1][1]) * b1 + ... - delta >= 0
            // ...
            // b0 >= 0
            // b1 >= 0
            // ...
            // b0 + b1 + ... + bn = 1.0

            // Goal: minimize delta (since it is on our side of the equation).
            {
                REAL one = 1.0;
                set_obj_fnex(lp, 1, &one, &cols);
                set_minim(lp);
            }

            set_add_rowmode(lp, TRUE);
            {
                REAL beliefRow = 1.0;

                // Each belief element needs to be greater or equal to zero (columns are 1-numbered)
                for ( int i = 1; i < states + 1; ++i )
                    add_constraintex(lp, 1, &beliefRow, &i, GE, 0.0);
            }

            // We will use this also to extract the solution.
            auto row   = new REAL[cols];
            {
                row[cols-1] = -1.0; // delta coefficient

                for ( auto & b : best ) {
                    for ( int i = 0; i < states; ++i )
                        row[i] = v[i] - b[i];
                    add_constraintex(lp, cols, row, NULL, GE, 0.0);
                }
                // Final simplex constraint
                for ( int i = 0; i < states; ++i )
                    row[i] = 1.0;
                add_constraintex(lp, states, row, NULL, EQ, 1.0);
            }

            auto result = solve(lp);

            if ( result ) throw 5;

            get_variables(lp, row);

            delete_lp(lp);

            // We have found a witness point if we have found a belief for which the value
            // of the supplied ValueFunction is greater than ALL others. Thus we just need
            // to verify that the variable we have minimized is actually less than 0.

            if ( row[cols-1] >= 0.0 )
                return std::make_pair(false, Belief());

            Belief solution(row, row + states);
            return std::make_pair(true, solution);
        }
    }
}
