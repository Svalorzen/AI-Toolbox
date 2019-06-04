#ifndef AI_TOOLBOX_FACTORED_FASTER_TRIE_HEADER_FILE
#define AI_TOOLBOX_FACTORED_FASTER_TRIE_HEADER_FILE

#include <random>
#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This class is a generally faster implementation of a Trie.
     *
     * This class stores keys in a different way from Trie, which allows it to
     * be much faster when retrieving. On the other hand, it is slightly less
     * flexible in what it can do.
     */
    class FasterTrie {
        public:
            using Entry = std::pair<size_t, PartialFactors>;
            using Entries = std::vector<Entry>;

            /**
             * @brief Basic constructor.
             *
             * This constructor simply copies the input state space and
             * uses it as bound to construct its internal data structures.
             *
             * @param f The factored space.
             */
            FasterTrie(Factors f);

            /**
             * @brief This function inserts a new id using the input as a key.
             *
             * Differently from Trie, we don't store the keys in an ordered
             * way, so this operation takes constant time (bar reallocations).
             *
             * @param pf The PartialFactors used as key for the insertion.
             *
             * @return The id of the newly inserted key.
             */
            size_t insert(PartialFactors pf);

            /**
             * @brief This function erases the id with the input key.
             *
             * This operation takes an amount of time proportional to the
             * number of keys with the same first element (key,value) of the
             * input PartialFactors.
             *
             * @param id The id to remove.
             * @param pf The PartialFactors used as key for the insertion.
             */
            void erase(size_t id, const PartialFactors & pf);

            /**
             * @brief This function returns all ids of the keys that match the input Factors.
             *
             * The output is not sorted.
             *
             * The input can have fewer elements than the space; the output
             * will be matched on those elements. Differently from Trie, it's
             * not possible to provide an offset.
             *
             * @param f The Factors to match against.
             *
             * @return The unsorted ids of all keys which match the input.
             */
            std::vector<size_t> filter(const Factors & f) const;

            /**
             * @brief This function returns a set of Entries which match the input and each other.
             *
             * The output set is constructed randomly to avoid bias. The output
             * of this function is thus randomized and not deterministic.
             *
             * We additionally return the Factors constructed by merging all
             * matches together. Any elements who couldn't be filled will be
             * set as the value of their space.
             *
             * @param pf The PartialFactors to match against.
             * @param remove Whether the matches should be removed from the FasterTrie.
             *
             * @return A set of Entry that match the input and each other, the Factors obtained by combining the input with the returned set.
             */
            std::tuple<Entries, Factors> reconstruct(const PartialFactors & pf, bool remove = false);

            /**
             * @brief This function returns the number of keys in the FasterTrie.
             *
             * @return The size of the FasterTrie.
             */
            size_t size() const;

            /**
             * @brief This function returns a reference of the internal Factors space.
             */
            const Factors & getF() const;

        private:
            Factors F;
            size_t counter_;

            std::vector<std::vector<Entries>> keys_;

            mutable std::ranlux24_base rand_; // Fastest engine possible, don't care about quality
            mutable std::vector<std::vector<size_t>> orders_;
    };
}

#endif
