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
    class SubsetEnumerator {
        public:
            using IdsStorage = std::vector<size_t>;

            /**
             * @brief Default constructor.
             *
             * @param elementsN The number of elements that the subset should have (<= limit);
             * @param limit The upper bound for each element in the subset (excluded).
             */
            SubsetEnumerator(size_t elementsN, size_t limit);

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
            size_t advance();

            /**
             * @brief This function returns whether there are more subsets to be enumerated.
             */
            bool isValid() const;

            /**
             * @brief This function resets the enumerator to the valid beginning.
             */
            void reset();

            /**
             * @brief This function returns the number of subsets enumerated over.
             */
            unsigned subsetsSize() const;

            /**
             * @brief This function returns the size of the range covered.
             */
            size_t size() const;

            /**
             * @brief This operator returns the current combination.
             *
             * This operator can be called only if isValid() is true.
             * Otherwise behavior is undefined.
             *
             * @return The current combination.
             */
            const IdsStorage& operator*() const;

            /**
             * @brief This operator returns a pointer to the current combination.
             *
             * This operator can be called only if isValid() is true.
             * Otherwise behavior is undefined.
             *
             * @return The current combination.
             */
            const IdsStorage* operator->() const;

        private:
            size_t limit_;
            IdsStorage ids_;
    };

    /**
     * @brief This class enumerates over all possible subsets of a container.
     *
     * This class iterates over all possible subsets of elements in a
     * container. For each subset, it behaves as an iterable range over the
     * elements of the current subset.
     *
     * Once advanced, all previous iterators are invalidated.
     *
     * @tparam Container The type of the container to be iterated over.
     */
    template <typename Container>
    class SubsetMap {
        public:
            using value_type = typename Container::value_type;

            using iterator       = IndexMapIterator<decltype(std::declval<SubsetEnumerator>()->begin()),  Container>;
            using const_iterator = IndexMapIterator<decltype(std::declval<SubsetEnumerator>()->cbegin()), const Container>;

            /**
             * @brief Default constructor.
             *
             * @param elementsN The number of elements that the subset should have (<= items.size()).
             * @param items The items container to be iterated on.
             */
            SubsetMap(size_t elementsN, Container & items) :
                    ids_(elementsN, items.size()), items_(items) {}

            /**
             * @brief This function advances the SubsetMap to the next possible subset.
             *
             * \sa SubsetEnumerator::advance()
             */
            auto advance() {
                return ids_.advance();
            }

            /**
             * @brief This function returns whether there are more subsets to be enumerated.
             */
            bool isValid() const {
                return ids_.isValid();
            }

            /**
             * @brief This function returns an iterator to the beginning of the current subset.
             */
            auto begin() { return iterator(ids_->begin(), items_); }

            /**
             * @brief This function returns a const_iterator to the beginning of the current subset.
             */
            auto begin() const { return cbegin(); }

            /**
             * @brief This function returns a const_iterator to the beginning of the current subset.
             */
            auto cbegin() const { return const_iterator(ids_->cbegin(), items_); }

            /**
             * @brief This function returns an iterator to the end of the current subset.
             */
            auto end() { return iterator(ids_->end(), items_); };

            /**
             * @brief This function returns a const_iterator to the end of the current subset.
             */
            auto end() const { return cend(); }

            /**
             * @brief This function returns a const_iterator to the end of the current subset.
             */
            auto cend() const { return const_iterator(ids_->cend(), items_); }

            /**
             * @brief This function returns the size of the range covered.
             */
            auto size() const { return ids_.size(); }

            /**
             * @brief This function returns the enumerator over the subsets.
             */
            const SubsetEnumerator & getEnumerator() const { return ids_; }

        private:
            SubsetEnumerator ids_;
            Container & items_;
    };
}
