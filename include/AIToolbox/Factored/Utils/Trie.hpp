#ifndef AI_TOOLBOX_FACTORED_TRIE_HEADER_FILE
#define AI_TOOLBOX_FACTORED_TRIE_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Utils/IndexMap.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This class organizes data ids as if in a trie.
     *
     * This class implements a trie, which is a kind of tree that can be
     * used to sort strings, or in our case partial states. This class
     * tries to be as efficient as possible, with tradeoffs for space and
     * time.
     *
     * Adding automatically inserts an id one greater than the last as value
     * within the trie, using the specified partial state as key.
     *
     * This data structure can then be filtered by Factors, and it will
     * match the Factors against all the PartialFactors that completely
     * match it.
     */
    class Trie {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor simply copies the input state space and
             * uses it as bound to construct its internal data structures.
             *
             * @param F The factored space.
             */
            Trie(Factors F);

            /**
             * @brief This function returns the set state space for the Trie.
             *
             * @return The set state space.
             */
            Factors getF() const;

            /**
             * @brief This function reserves memory for at least size elements.
             *
             * It is recommended to call this function if there is a need to
             * insert very many elements to it. This will prevent multiple
             * reallocations after each insertion.
             *
             * @param size The number of elements to be reserved.
             */
            void reserve(size_t size);

            /**
             * @brief This function inserts a new id using the input as a key.
             *
             * If possible, try to insert keys from smallest to highest,
             * where the ordering is done by the sum of all the partial
             * states values, where unspecified states count as one over
             * the max of their possible value.
             *
             * This is because the underlying container is a vector, and
             * elements are arranged in numerical order, with unspecified
             * elements at the end. Inserting lower numbered elements before
             * guarantees minimal re-copying within the vectors which allows
             * for fast insertion.
             *
             * @param pf The PartialFactors used as key for the insertion.
             *
             * @return The id of the newly inserted key.
             */
            size_t insert(const PartialFactors & pf);

            /**
             * @brief This function returns the number of insertions performed on the Trie.
             *
             * @return The number of insertions done on the tree.
             */
            size_t size() const;

            /**
             * @brief This function returns all ids where their key matches the input Factors.
             *
             * This function is the core of this data structure. For each
             * factor of the input Factors, it maintains a list of
             * all ids which could match it at that factor. It then
             * performs an intersection between all these list, starting
             * from the smaller ones to the larger ones in order to perform
             * the minimum number of comparisons possible.
             *
             * If an id is found matching in all factors, then such a key
             * was inserted and is added to the returned list.
             *
             * This function can also be used to filter on Factors smaller
             * than the real one, as long as they are all adjacent. The
             * offset parameter can be used to specify by how many factors
             * to offset the input. For example, an input of {3,5} with
             * offset 0 will look for all inserted PartialFactors with 3 at
             * position 0, and 5 at position 1. The same input with offset
             * 2 will look for all inserted PartialFactors with 3 at
             * position 2, and 5 at position 3.
             *
             * @param f The Factors used as filter in the trie.
             * @param offset The offset for each factor in the input.
             *
             * @return The ids of all inserted keys which match the input.
             */
            std::vector<size_t> filter(const Factors & f, size_t offset = 0) const;

            /**
             * @brief This function returns all ids where their key matches the input Factors.
             *
             * This function is the core of this data structure. For each
             * factor of the input PartialFactors, it maintains a list of
             * all ids which could match it at that factor. It then
             * performs an intersection between all these list, starting
             * from the smaller ones to the larger ones in order to perform
             * the minimum number of comparisons possible.
             *
             * If an id is found matching in all factors, then such a key
             * was inserted and is added to the returned list.
             *
             * @param f The PartialFactors used as filter in the trie.
             *
             * @return The ids of all inserted keys which match the input.
             */
            std::vector<size_t> filter(const PartialFactors & pf) const;

            /**
             * @brief This function refines the input ids with the supplied filter.
             *
             * @param ids The ids to consider for filtering.
             * @param pf The PartialFactors used as a filter in the trie.
             *
             * @return The ids in the input which match the filter.
             */
            std::vector<size_t> refine(const std::vector<size_t> & ids, const PartialFactors & pf) const;

            /**
             * @brief This function removes the input id from the trie.
             *
             * Note that this function performs a lookup on all vectors whether
             * the id is really present or not (maybe because you erased it
             * before).
             *
             * @param id The id to remove.
             */
            void erase(size_t id);

            /**
             * @brief This function removes the input id from the trie.
             *
             * This function is faster than erase(size_t) as it already knows what to look for.
             *
             * Note that this function performs a lookup on all vectors whether
             * the id is really present or not (maybe because you erased it
             * before).
             *
             * @param id The id to remove.
             * @param pf The key with which the id was inserted.
             */
            void erase(size_t id, const PartialFactors & pf);

            /**
             * @brief This function returns a reference to the underlying Factors.
             *
             * @return The Factors domain of this Trie.
             */
            const Factors & getFactors() const;

        private:
            /**
             * @brief This function returns all ids currently in the Trie.
             *
             * @return The ids in the Trie.
             */
            std::vector<size_t> getAllIds() const;

            Factors F;
            size_t counter_;

            std::vector<std::vector<std::vector<size_t>>> ids_;
    };
}

#endif
