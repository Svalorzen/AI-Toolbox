#ifndef AI_TOOLBOX_FACTORED_CPS_QUEUE_HEADER_FILE
#define AI_TOOLBOX_FACTORED_CPS_QUEUE_HEADER_FILE

#include <random>
#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This class is a not-general-purpose fast implementation of a Trie.
     *
     * This class assumes keys are always the parent sets of some variable as
     * represented in a FactoredDDN. It works similarly as a FasterTrie, but
     * this knowledge allows it to be more efficient in its bucket management.
     *
     * We also assume there are no duplicate entries!
     */
    class CPSQueue {
        public:
            struct N {
                PartialKeys tag;
                Vector priorities;
                double maxV;
                size_t maxS;
            };
            struct Node {
                PartialKeys actionTag;
                double maxV;
                size_t maxA;
                std::vector<size_t> order;
                std::vector<N> nodes;
            };

            /**
             * @brief Basic constructor.
             *
             * This constructor simply copies the input state space and
             * uses it as bound to construct its internal data structures.
             *
             * @param s The factored state space.
             */
            CPSQueue(State S, Action A, const FactoredDDN & ddn);

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
            void update(size_t i, size_t a, size_t s, double p);

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
            // void erase(size_t i, const PartialValues & s, const PartialValues & a);

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
            // std::vector<size_t> filter(const Factors & f) const;

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
            std::tuple<State, Action> reconstruct();

        private:
            State S;
            Action A;

            std::vector<size_t> order_;
            std::vector<Node> nodes_;

            // FIXME: remove?
            mutable std::ranlux24_base rand_; // Fastest engine possible, don't care about quality
    };
}

#endif
