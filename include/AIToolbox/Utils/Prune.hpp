#ifndef AI_TOOLBOX_UTILS_PRUNE_HEADER_FILE
#define AI_TOOLBOX_UTILS_PRUNE_HEADER_FILE

#include <algorithm>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/Core.hpp>

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
     * @param end The end of the list that needs to be pruned.
     *
     * @return The iterator that separates dominated elements with non-pruned.
     */
    template <typename Iterator>
    Iterator extractDominated(const size_t N, Iterator begin, Iterator end) {
        if ( std::distance(begin, end) < 2 ) return end;

        auto dominates = [N](auto lhs, auto rhs) {
            for ( size_t i = 0; i < N; ++i )
                if ( rhs[i] > lhs[i] ) return false;
            return true;
        };

        auto optEnd = begin, target = end - 1;
        while (optEnd < end) {
            target = end - 1; // The one we are checking whether it is dominated.
            // Check against proven non-dominated vectors
            for (auto iter = begin; iter < optEnd; ++iter) {
                if (dominates(*iter, *target)) {
                    --end;
                    goto next;
                }
            }
            {
                // Check against others and find another non-dominated. Here
                // we go from the back so that we only swap with vectors we
                // have already checked.
                auto helper = target;
                while (helper != optEnd) {
                    --helper;
                    // If dominated, remove it and continue from there.
                    if (dominates(*helper, *target)) {
                        std::iter_swap(baseIter(target), baseIter(--end));
                        target = helper;
                    }
                }
                // Add vector we found in the non-dominated group
                std::iter_swap(baseIter(target), baseIter(optEnd));
                ++optEnd;
            }
next:;
        }
        return end;
    }
}

#endif
