#ifndef AI_TOOLBOX_FACTORED_FILTER_MAP_HEADER_FILE
#define AI_TOOLBOX_FACTORED_FILTER_MAP_HEADER_FILE

#include <AIToolbox/Utils/IndexMap.hpp>
#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Trie.hpp>
#include <AIToolbox/Factored/Utils/FasterTrie.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This class is a container which uses PartialFactors as keys.
     *
     * This class stores values using PartialFactors as keys. The values can
     * then be reached using Factors. The result will be an iterable object
     * which will iterate over all values where the key matched the input.
     *
     * This class does not allow removal of elements.
     *
     * @tparam T The type of object to be stored.
     */
    template <typename T, typename TrieType = FasterTrie>
    class FilterMap {
        public:
            using ItemsContainer = std::vector<T>;
            using Iterable = IndexMap<std::vector<size_t>, ItemsContainer>;
            using ConstIterable = IndexMap<std::vector<size_t>, const ItemsContainer>;

            /**
             * @brief Basic constructor.
             *
             * This constructor simply initializes the underlying TrieType
             * object with the input factor space.
             *
             * @param f The desired factor space.
             */
            FilterMap(Factors f) : ids_(std::move(f)) {}

            /**
             * @brief Constructor from trie and items.
             *
             * This constructor is provided when the user wants to copy two
             * FilterMap of different types but which share the underlying
             * factorization.
             *
             * With this constructor, the underlying TrieType can be copied,
             * while a new container of items must be provided, of the same
             * size as the input TrieType.
             *
             * If the two sizes are not equal, the constructor will throw
             * an std::invalid_argument exception.
             *
             * @param t The trie to copy.
             * @param c The new items to store.
             */
            FilterMap(TrieType t, ItemsContainer c) :
                    ids_(std::move(t)), items_(std::move(c))
            {
                if (ids_.size() != items_.size())
                    throw std::invalid_argument("Input trie and container have different sizes!");
            }

            /**
             * @brief This function returns the set factor space for the FilterMap.
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
             * Note that the input may be shorter than the overall Factor
             * domain. In any case, we assume the search will begin from the
             * zero element.
             *
             * @param f The key that must be matched.
             *
             * @return An iterable object over all values matching the input.
             */
            Iterable filter(const Factors & f) {
                return Iterable(ids_.filter(f), items_);
            }

            /**
             * @brief This function creates an iterable object over all values matching the input key.
             *
             * Note that the input may be shorter than the overall Factor
             * domain. In any case, we assume the search will begin from the
             * zero element.
             *
             * @param f The key that must be matched.
             * @param offset The offset of the key, if smaller than the factor space.
             *
             * @return An iterable object over all values matching the input.
             */
            ConstIterable filter(const Factors & f) const {
                return ConstIterable(ids_.filter(f), items_);
            }

            /**
             * @brief This function creates an iterable object over all values matching the input key.
             *
             * \sa Trie::filter(const Factors&, size_t)
             *
             * This method can only be used if the underlying TrieType supports it.
             *
             * @param f The key that must be matched.
             * @param offset The offset of the key, if smaller than the factor space.
             *
             * @return An iterable object over all values matching the input.
             */
            template <typename TT = TrieType, typename = std::enable_if_t<std::is_same_v<TT, Trie>>>
            Iterable filter(const Factors & f, size_t offset) {
                return Iterable(ids_.filter(f, offset), items_);
            }

            /**
             * @brief This function creates an iterable object over all values matching the input key.
             *
             * \sa Trie::filter(const Factors&, size_t)
             *
             * This method can only be used if the underlying TrieType supports it.
             *
             * @param f The key that must be matched.
             * @param offset The offset of the key, if smaller than the factor space.
             *
             * @return An iterable object over all values matching the input.
             */
            template <typename TT = TrieType, typename = std::enable_if_t<std::is_same_v<TT, Trie>>>
            ConstIterable filter(const Factors & f, size_t offset) const {
                return ConstIterable(ids_.filter(f, offset), items_);
            }

            /**
             * @brief This function creates an iterable object over all values matching the input key.
             *
             * \sa Trie::filter(const PartialFactors&)
             *
             * This method can only be used if the underlying TrieType supports it.
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
             * This method can only be used if the underlying TrieType supports it.
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
                if constexpr (std::is_same_v<TrieType, Trie>)
                    ids_.reserve(size);

                items_.reserve(size);
            }

            /**
             * @brief This function returns the number of values that have been added to the FilterMap.
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
             * @brief This function returns the underlying trie object.
             *
             * This function is provided in case one wants to manually
             * access and call the underlying trie.
             *
             * @return The trie associated with this container.
             */
            const TrieType & getTrie() const {
                return ids_;
            }

        private:
            TrieType ids_;
            ItemsContainer items_;
    };
}

#endif
