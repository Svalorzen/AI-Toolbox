#ifndef AI_TOOLBOX_POMDP_PRUNER_HEADER_FILE
#define AI_TOOLBOX_POMDP_PRUNER_HEADER_FILE

#include <utility>

#include <AIToolbox/POMDP/Types.hpp>

#include <lpsolve/lp_types.h>

namespace AIToolbox {
    namespace POMDP {
        /**
         * @brief This class offers pruning facilities for non-parsimonious ValueFunction sets.
         */
        class Pruner {
            public:
                Pruner(size_t S);

                /**
                 * @brief This function prunes all non useful ValueFunctions from the provided VList.
                 *
                 * @param w The list that needs to be pruned.
                 */
                void operator()(VList * w);

                /**
                 * @brief This function prunes all ValueFunctions in the VList that are dominated by others.
                 *
                 * This function performs simple comparisons between all ValueFunctions in the VList,
                 * and is thus much more performant than the prune() function, since that needs to solve
                 * multiple linear programming problems. However, this function will not return the truly
                 * parsimonious set of ValueFunctions, as its pruning powers are limited.
                 *
                 * @param S The number of states in the Model.
                 * @param pw The list that needs to be pruned.
                 */
                static void dominationPrune(size_t S, VList * pw);

                /**
                 * @brief This function finds and moves all best ValueFunctions in the simplex corners at the end of the specified range.
                 *
                 * What this function does is to find out which ValueFunctions give the highest value in
                 * corner beliefs. Since multiple corners may use the same ValueFunction, the number of
                 * found ValueFunctions may not be the same as the number of corners.
                 *
                 * This function uses an already existing bound containing previously marked useful
                 * ValueFunctions. The order is 'begin'->'bound'->'end', where bound may be equal to end
                 * where no previous bound exists. All found ValueFunctions are added between 'bound' and
                 * 'end', but only if they were not there previously.
                 *
                 * @param begin The begin of the search range.
                 * @param bound The begin of the 'useful' range.
                 * @param end The end of the search range. It is NOT included in the search.
                 *
                 * @return The new bound iterator.
                 */
                VList::iterator extractBestAtSimplexCorners(VList::iterator begin, VList::iterator bound, VList::iterator end);

                /**
                 * @brief This function finds a witness point where the ValueFunction provided is better than any ValueFunction in the VList.
                 *
                 * @param v The ValueFunction that may have a witness point.
                 * @param best The set of ValueFunctions that may or may not be optimal with respect to v.
                 *
                 * @return A pair containing true and the found witness point if such a witness exists, and false and empty vector otherwise.
                 */
                std::pair<bool, Belief> findWitnessPoint(const MDP::Values & v, const VList & best);

            private:
                size_t S;

                // LP_SOLVE DATA
                int cols;
                std::unique_ptr<lprec, void(*)(lprec*)> lp;
                std::unique_ptr<REAL[]> row;

                void setLP(size_t rows);
                void addRow(const MDP::Values & v, int constrType);
                void popRow();
        };
    }
}

#endif
