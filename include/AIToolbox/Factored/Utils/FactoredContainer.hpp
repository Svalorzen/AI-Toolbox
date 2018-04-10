#ifndef AI_TOOLBOX_FACTORED_CONTAINER_HEADER_FILE
#define AI_TOOLBOX_FACTORED_CONTAINER_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This class organizes data ids as if in a trie.
     *
     * This class implements a trie, which is a kind of tree that can be
     * used to sort strings, or in our case partial states. This class
     * tries to be as efficient as possible, with tradeoffs for space and
     * time.
     *
     * Currently this implementation only supports adding. Adding
     * automatically inserts an id one greater than the last as value
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
             * @param s The factored state space.
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
             * @param ps The partial state used as key for the insertion.
             */
            void insert(const PartialFactors & pf);

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
             * @param f The Factors used as filter in the trie.
             *
             * @return The ids of all inserted keys which match the input.
             */
            std::vector<size_t> filter(const PartialFactors & pf) const;

        private:
            Factors F;
            size_t counter_;

            std::vector<std::vector<size_t>> partials_;
            std::vector<std::vector<size_t>> ids_;
    };

    template <typename Container>
    class FactoredIterable;

    /**
     * @brief This class is a container which uses PartialFactors as keys.
     *
     * This class stores values using PartialFactors as keys. The values
     * can then be reached using Factors. The result will be an
     * iterable object which will iterate over all values where the
     * key matched the input.
     *
     * @tparam T The type of object to be stored.
     */
    template <typename T>
    class FactoredContainer {
        public:
            using ItemsContainer = std::vector<T>;
            using Iterable = FactoredIterable<ItemsContainer>;
            using ConstIterable = FactoredIterable<const ItemsContainer>;

            /**
             * @brief Basic constructor.
             *
             * This constructor simply initializes the underlying Trie
             * with the input factor space.
             *
             * @param f The desired factor space.
             */
            FactoredContainer(Factors f) : ids_(std::move(f)) {}

            /**
             * @brief Constructor from Trie and items.
             *
             * This constructor is provided when the user wants to copy two
             * FactoredContainer of different types but which share the
             * underlying factorization.
             *
             * With this constructor, the underlying Trie can be copied,
             * while a new container of items must be provided, of the same
             * size as the input Trie.
             *
             * If the two sizes are not equal, the constructor will throw
             * an std::invalid_argument exception.
             *
             * @param t The trie to copy.
             * @param c The new items to store.
             */
            FactoredContainer(Trie t, ItemsContainer c) :
                    ids_(std::move(t)), items_(std::move(c))
            {
                if (ids_.size() != items_.size())
                    throw std::invalid_argument("Input trie and container have different sizes!");
            }

            /**
             * @brief This function returns the set factor space for the FactoredContainer.
             *
             * @return The set factor space.
             */
            Factors getF() const {
                return ids_.getF();
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
            void emplace(const PartialFactors & pf, Args&&... args) {
                ids_.insert(pf);
                items_.emplace_back(std::forward<Args>(args)...);
            }

            /**
             * @brief This function creates an iterable object over all values matching the input key.
             *
             * \sa Trie::filter(const Factors&, size_t)
             *
             * @param f The key that must be matched.
             * @param offset The offset of the key, if smaller than the factor space.
             *
             * @return An iterable object over all values matching the input.
             */
            Iterable filter(const Factors & f, size_t offset = 0) {
                return Iterable(ids_.filter(f, offset), items_);
            }

            /**
             * @brief This function creates an iterable object over all values matching the input key.
             *
             * \sa Trie::filter(const PartialFactors&)
             *
             * @param f The key that must be matched.
             * @param offset The offset of the key, if smaller than the factor space.
             *
             * @return An iterable object over all values matching the input.
             */
            ConstIterable filter(const Factors & f, size_t offset = 0) const {
                return ConstIterable(ids_.filter(f, offset), items_);
            }

            /**
             * @brief This function creates an iterable object over all values matching the input key.
             *
             * \sa Trie::filter(const PartialFactors&)
             *
             * @param s The key that must be matched.
             *
             * @return An iterable object over all values matching the input.
             */
            Iterable filter(const PartialFactors & pf) {
                return Iterable(ids_.filter(pf), items_);
            }

            /**
             * @brief This function creates an iterable object over all values matching the input key.
             *
             * \sa Trie::filter(const PartialFactors&)
             *
             * @param s The key that must be matched.
             *
             * @return An iterable object over all values matching the input.
             */
            ConstIterable filter(const PartialFactors & pf) const {
                return ConstIterable(ids_.filter(pf), items_);
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

            /**
             * @brief This function returns the beginning of a range containing all items added to the container.
             *
             * @return The beginning of the range of all items.
             */
            typename ItemsContainer::iterator begin() {
                return items_.begin();
            }

            /**
             * @brief This function returns the beginning of a range containing all items added to the container.
             *
             * @return The beginning of the range of all items.
             */
            typename ItemsContainer::const_iterator begin() const {
                return items_.begin();
            }

            /**
             * @brief This function returns the end of a range containing all items added to the container.
             *
             * @return The end of the range of all items.
             */
            typename ItemsContainer::iterator end() {
                return items_.end();
            }

            /**
             * @brief This function returns the end of a range containing all items added to the container.
             *
             * @return The end of the range of all items.
             */
            typename ItemsContainer::const_iterator end() const {
                return items_.end();
            }

            /**
             * @brief This function allows direct access to the items in the container.
             *
             * This function is provided in case one wants to manually
             * access and call the underlying Trie.
             *
             * No bound checking is performed.
             *
             * @param id The id of the item.
             *
             * @return A constant reference to the accessed item.
             */
            const T & operator[](size_t id) const {
                return items_[id];
            }

            /**
             * @brief This function allows direct access to the items in the container.
             *
             * This function is provided in case one wants to manually
             * access and call the underlying Trie.
             *
             * No bound checking is performed.
             *
             * @param id The id of the item.
             *
             * @return A reference to the accessed item.
             */
            T & operator[](size_t id) {
                return items_[id];
            }

            /**
             * @brief This function provides a direct view on the items contained by the container.
             *
             * @return The underlying data container.
             */
            const ItemsContainer & getContainer() const {
                return items_;
            }

            /**
             * @brief This function returns the underlying Trie object.
             *
             * This is useful since it is the Trie which does all the heavy
             * lifting in this class (filtering).
             *
             * @return The Trie associated with this container.
             */
            const Trie & getTrie() const {
                return ids_;
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
    template <typename Container>
    class FactoredIterable {
        public:
            template <typename T>
            class FactoredIterator;

            using value_type = typename Container::value_type;
            using iterator = FactoredIterator<typename copy_const<value_type, Container>::type>;
            using const_iterator = FactoredIterator<const value_type>;

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
            FactoredIterable(std::vector<size_t> ids, Container & items) : ids_(std::move(ids)), items_(items) {}

            /**
             * @brief This function returns an iterator to the beginning of this filtered range.
             */
            iterator begin() { return ids_.size() ? iterator(this) : iterator(); }

            /**
             * @brief This function returns a const_iterator to the beginning of this filtered range.
             */
            const_iterator begin() const { return ids_.size() ? const_iterator(this) : const_iterator(); }

            /**
             * @brief This function returns an iterator to the end of this filtered range.
             */
            iterator end() { return iterator(); };

            /**
             * @brief This function returns a const_iterator to the end of this filtered range.
             */
            const_iterator end() const { return const_iterator(); }

            /**
             * @brief This function returns the size of the range covered.
             */
            size_t size() const { return ids_.size(); }

        private:
            friend iterator;
            friend const_iterator;

            const std::vector<size_t> ids_;
            Container & items_;
    };

    /**
     * @brief This class is a simple iterator to iterate over filtered values held in a FactoredIterable.
     */
    template <typename Container>
    template <typename T>
    class FactoredIterable<Container>::FactoredIterator {
        private:
            using Encloser = typename copy_const<FactoredIterable<Container>, T>::type;
        public:
            using value_type = T;

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

            value_type& operator*()  { return parent_->items_[parent_->ids_[currentId_]]; }
            value_type* operator->() { return &(operator*()); }

            /**
             * @brief This function returns the equivalent item id of this iterator in its container.
             */
            size_t toContainerId() const { return parent_->ids_[currentId_]; }

            void operator++() {
                ++currentId_;
                if ( currentId_ >= parent_->ids_.size() ) {
                    currentId_ = 0;
                    parent_ = nullptr;
                }
            }

            bool operator==(const FactoredIterator & other) {
                if ( parent_ == other.parent_ ) return currentId_ == other.currentId_;
                return false;
            }
            bool operator!=(const FactoredIterator & other) { return !(*this == other); }

        private:
            size_t currentId_;
            Encloser * parent_;
    };
}

#endif
