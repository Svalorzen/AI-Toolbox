#ifndef AI_TOOLBOX_UTILS_PRUNE_HEADER_FILE
#define AI_TOOLBOX_UTILS_PRUNE_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <algorithm>

namespace AIToolbox {
    /**
     * @brief This function finds and moves all Vectors in the range that are dominated by others.
     *
     * This function performs simple comparisons between all Vectors in the
     * range, and is thus much more performant than a full-fledged prune, since
     * that would need to solve multiple linear programming problems. However,
     * this function will not return the truly parsimonious set of
     * Vectors, as its pruning powers are limited.
     *
     * Dominated elements will be moved at the end of the range for safe removal.
     *
     * @param N The number of elements in each Vector.
     * @param begin The begin of the list that needs to be pruned.
     * @param begin The end of the list that needs to be pruned.
     *
     * @return The iterator that separates dominated elements with non-pruned.
     */
    template <typename Iterator>
    Iterator extractDominated(const size_t N, Iterator begin, Iterator end) {
        if ( std::distance(begin, end) < 2 ) return end;

        // We use this comparison operator to filter all dominated vectors.
        // We define a vector to be dominated by an equal vector, so that
        // we can remove duplicates in a single swoop. However, we avoid
        // removing everything by returning false for comparison of the vector with itself.
        struct {
            const Vector * rhs;
            size_t N;
            bool operator()(const Vector & lhs) {
                if ( &lhs == rhs ) return false;
                for ( size_t i = 0; i < N; ++i )
                    if ( (*rhs)[i] > lhs[i] ) return false;
                return true;
            }
        } dominates;

        dominates.N = N;

        // For each vector, if we find a vector that dominates it, then we remove it.
        // Otherwise we continue, comparing every vector with every other non-dominated
        // vector.
        Iterator iter = begin, helper;
        while ( iter < end ) {
            dominates.rhs = &(*iter);
            helper = std::find_if(begin, end, dominates);
            if ( helper != end )
                std::iter_swap( iter, --end );
            else
                ++iter;
        }
        return end;
    }
}

#endif
