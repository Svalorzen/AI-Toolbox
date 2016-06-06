#ifndef FACTORED_CONTAINER_HEADER_FILE
#define FACTORED_CONTAINER_HEADER_FILE

#include <AIToolbox/FactoredMDP/Types.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        /**
         * @brief This class organizes data ids as if in a trie.
         *
         * This class implements a trie, which is a kind of tree that can be
         * used to sort strings, or in our case partial states. This class
         * tries to be as efficient as possible, with tradeoffs for space and
         * time.
         *
         * Currently this implementation only supports adding. Adding
         * automatically inserts and id one greater than the last as value
         * within the trie, using the specified partial state as key.
         *
         * This data structure can then be filtered by full states, and it will
         * match the full states against all the partial states that completely
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
                 * @param s The factored state space.
                 */
                Trie(State S);

                /**
                 * @brief This function returns the set state space for the Trie.
                 *
                 * @return The set state space.
                 */
                State getS() const;

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
                 * @param ps The partial state used as key for the insertion.
                 */
                void insert(const PartialState & ps);

                /**
                 * @brief This function returns the number of insertions performed on the Trie.
                 *
                 * @return The number of insertions done on the tree.
                 */
                size_t size() const;

                /**
                 * @brief This function returns all ids where their key matches the input State.
                 *
                 * This function is the core of this data structure. For each
                 * factor of the input State, it maintains a list of
                 * all ids which could match it at that factor. It then
                 * performs an intersection between all these list, starting
                 * from the smaller ones to the larger ones in order to perform
                 * the minimum number of comparisons possible.
                 *
                 * If an id is found matching in all factors, then such a key
                 * was inserted and is added to the returned list.
                 *
                 * @param s The State used as filter in the trie.
                 *
                 * @return The ids of all inserted keys which match the input.
                 */
                std::vector<size_t> filter(const State & s) const;

            public:
                State S;
                size_t counter_;

                std::vector<std::vector<size_t>> partials_;
                std::vector<std::vector<size_t>> ids_;
        };

        template <typename FactoredContainer>
        class FactoredIterable;

        /**
         * @brief This class is a container which uses PartialStates as keys.
         *
         * This class stores values using PartialStates as keys. The values
         * can then be reached using States. The result will be an
         * iterable object which will iterate over all values where the
         * key matched the input.
         *
         * @tparam T The type of object to be stored.
         */
        template <typename T>
        class FactoredContainer {
            public:
                using ItemsContainer = std::vector<T>;
                using Iterable = FactoredIterable<FactoredContainer<T>>;

                /**
                 * @brief Basic constructor.
                 *
                 * This constructor simply initializes the underlying Trie
                 * with the input state space.
                 *
                 * @param s The environment state space.
                 */
                FactoredContainer(State s) : ids_(s) {}

                /**
                 * @brief This function returns the set state space for the FactoredContainer.
                 *
                 * @return The set state space.
                 */
                State getS() const {
                    return ids_.getS();
                }

                /**
                 * @brief This function creates an instance of T within the container.
                 *
                 * If very many items must be emplaced in the container, it
                 * is recommended to call reserve() beforehand in order to
                 * avoid multiple reallocations. In addition, see the
                 * Trie documentation on how to most efficiently
                 * insert new keys (if speed is very important).
                 *
                 * \sa Trie::insert()
                 *
                 * @param ps The key where the value should be stored.
                 * @param args The arguments needed to emplace the new value.
                 */
                template <typename... Args>
                void emplace(const PartialState & ps, Args&&... args) {
                    ids_.insert(ps);
                    items_.emplace_back(std::forward<Args>(args)...);
                }

                /**
                 * @brief This function creates an iterable object over all values matching the input key.
                 *
                 * @param s The key that must be matched.
                 *
                 * @return An iterable object over all values matching the input.
                 */
                Iterable filter(const State & s) {
                    return Iterable(ids_.filter(s), items_);
                }

                /**
                 * @brief This function reserves the specified space to avoid reallocations.
                 *
                 * @param size The minimum number of elements we should reserve space for.
                 */
                void reserve(size_t size) {
                    ids_.reserve(size);
                    items_.reserve(size);
                }

                /**
                 * @brief This function returns the number of values that have been added to the FactoredContainer.
                 *
                 * @return The number of container values.
                 */
                size_t size() const {
                    return items_.size();
                }

            private:
                Trie ids_;
                ItemsContainer items_;
        };

        /**
         * @brief This class is an iterable construct on the filtered results over a FactoredContainer.
         *
         * @tparam FactoredContainer The type of the parent FactoredContainer.
         */
        template <typename FactoredContainer>
        class FactoredIterable {
            public:
                class FactoredIterator;
                using Container = typename FactoredContainer::ItemsContainer;

                using value_type = typename Container::value_type;
                using iterator = FactoredIterator;

                /**
                 * @brief Basic constructor.
                 *
                 * This constructor stores all the ids and items over which to
                 * iterate.
                 *
                 * Keep in mind that this object WILL be invalidated if the
                 * input item container is modified or destroyed.
                 *
                 * @param ids The ids to iterate over.
                 * @param items The items
                 */
                FactoredIterable(std::vector<size_t> ids, Container & items) : ids_(ids), items_(items) {}

                /**
                 * @brief This function returns an iterator to the beginning of this filtered range.
                 */
                iterator begin() { return iterator(this); }
                /**
                 * @brief This function returns an iterator to the end of this filtered range.
                 */
                iterator end() { return iterator(); };

            private:
                friend class FactoredIterator;
                const std::vector<size_t> ids_;
                Container & items_;
        };

        /**
         * @brief This class is a simple iterator to iterate over filtered values held in a FactoredIterable.
         */
        template <typename FactoredContainer>
        class FactoredIterable<FactoredContainer>::FactoredIterator {
            private:
                using Encloser = FactoredIterable<FactoredContainer>;
            public:
                using value_type = typename Encloser::value_type;

                /**
                 * @brief Basic constructor for end iterators.
                 */
                FactoredIterator() : currentId_(0), parent_(nullptr) {}

                /**
                 * @brief Basic constructor for begin iterators.
                 *
                 * @param parent The parent iterable object holding ids and values.
                 */
                FactoredIterator(Encloser * parent) : currentId_(0), parent_(parent) {}

                value_type& operator*() {                return parent_->items_[parent_->ids_[currentId_]]; }
                const value_type& operator*() const {    return parent_->items_[parent_->ids_[currentId_]]; }

                value_type* operator->() {               return &(operator*()); }
                const value_type* operator->() const {   return &(operator*()); }

                void operator++() {
                    ++currentId_;
                    if ( currentId_ >= parent_->ids_.size() ) {
                        currentId_ = 0;
                        parent_ = nullptr;
                    }
                }

                bool operator==(const FactoredIterator & other ) {
                    if ( parent_ == other.parent_ ) return currentId_ == other.currentId_;
                    return false;
                }
                bool operator!=(const FactoredIterator & other ) { return !(*this == other); }

            private:
                size_t currentId_;
                Encloser * parent_;
        };
    }
}

#endif
