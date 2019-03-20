#ifndef AI_TOOLBOX_UTILS_COMBINATORICS_HEADER_FILE
#define AI_TOOLBOX_UTILS_COMBINATORICS_HEADER_FILE

#include <AIToolbox/Utils/IndexMap.hpp>

namespace AIToolbox {
    /**
     * @brief Returns (n k); i.e. n choose k.
     */
    unsigned nChooseK(unsigned n, unsigned k);

    /**
     * @brief Returns the number of stars/bars combinations.
     *
     * This function returns the number of combinations for dividing a set of
     * stars with `bars` number of elements.
     */
    unsigned starsBars(unsigned stars, unsigned bars);

    /**
     * @brief Returns the number of balls/bins combinations.
     *
     * This function returns the number of combinations in which you can split
     * a set of indistinguishable balls into a set of distinguishable bins.
     *
     * This is equivalent to starsBars(balls, bins - 1).
     *
     * Note: bins shall NOT be equal to 0.
     */
    unsigned ballsBins(unsigned balls, unsigned bins);

    /**
     * @brief Returns the number of stars/bars combinations.
     *
     * This function returns the number of combinations for dividing a set of
     * stars with `bars` number of elements, where no two bars can be adjacent.
     */
    unsigned nonZeroStarsBars(unsigned stars, unsigned bars);

    /**
     * @brief Returns the number of balls/bins combinations.
     *
     * This function returns the number of combinations in which you can split
     * a set of indistinguishable balls into a set of distinguishable bins,
     * where no bin can be empty.
     *
     * This is equivalent to nonZeroStarsBars(balls, bins - 1).
     *
     * Note: bins shall NOT be equal to 0.
     */
    unsigned nonZeroBallsBins(unsigned balls, unsigned bins);

    /**
     * @brief This class enumerates all possible vectors of finite subsets over N elements.
     */
    template <typename Index>
    class SubsetEnumerator {
        public:
            using IdsStorage = std::vector<Index>;

            /**
             * @brief Default constructor.
             *
             * @param elementsN The number of elements that the subset should have (<= limit);
             * @param limit The upper bound for each element in the subset (excluded).
             */
            SubsetEnumerator(size_t elementsN, Index lowerBound, Index upperBound) :
                    lowerBound_(lowerBound), upperBound_(upperBound), ids_(elementsN)
            {
                assert(elementsN >= 0);
                if constexpr (std::is_integral_v<Index>)
                    assert(static_cast<size_t>(upperBound_ - lowerBound_) >= elementsN);
                else
                    assert(static_cast<size_t>(std::distance(lowerBound_, upperBound_)) >= elementsN);
                reset();
            }

            /**
             * @brief This function advances the SubsetEnumerator to the next possible subset.
             *
             * This function iterates first on the last elements of the subset
             * vector, and iterates over the previous ones once it reaches the
             * end. For example, for a subset of length 3 over 6 elements the
             * iteration will look like this:
             *
             * 0, 1, 2
             * 0, 1, 3
             * 0, 1, 4
             * 0, 1, 5
             * 0, 2, 3
             * 0, 2, 4
             * 0, 2, 5
             * 1, 2, 3
             * etc.
             *
             * The number returned by this function represents the id of the
             * leftmost (lowest) element that has been changed by the advance.
             * This may be useful in case you need to do some work for the
             * elements that changed in the subset, and want to lose as little
             * time as possible.
             *
             * @return The id of the leftmost element changed by the advance.
             */
            auto advance() {
                auto current = ids_.size() - 1;
                auto ub = upperBound_ - 1;
                while (current && ids_[current] == ub) --current, --ub;

                auto lowest = current; // Last element we need to change.
                ub = ++ids_[current];

                while (++current != ids_.size()) ids_[current] = ++ub;

                return lowest;
            }

            /**
             * @brief This function returns whether there are more subsets to be enumerated.
             */
            bool isValid() const {
                return ids_.back() < upperBound_;
            }

            /**
             * @brief This function resets the enumerator to the valid beginning.
             */
            void reset() {
                std::iota(std::begin(ids_), std::end(ids_), lowerBound_);
            }

            /**
             * @brief This function returns the number of subsets enumerated over.
             */
            auto subsetsSize() const {
                if constexpr (std::is_integral_v<Index>)
                    return nChooseK(upperBound_ - lowerBound_, ids_.size());
                else
                    return nChooseK(std::distance(lowerBound_, upperBound_), ids_.size());
            }

            /**
             * @brief This function returns the size of the range covered.
             */
            auto size() const { return ids_.size(); }


            /**
             * @brief This operator returns the current combination.
             *
             * This operator can be called only if isValid() is true.
             * Otherwise behavior is undefined.
             *
             * @return The current combination.
             */
            const IdsStorage& operator*() const {
                return ids_;
            }

            /**
             * @brief This operator returns a pointer to the current combination.
             *
             * This operator can be called only if isValid() is true.
             * Otherwise behavior is undefined.
             *
             * @return The current combination.
             */
            const IdsStorage* operator->() const {
                return &ids_;
            }

        private:
            Index lowerBound_, upperBound_;
            IdsStorage ids_;
    };
}

#endif
