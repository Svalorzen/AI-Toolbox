#include <AIToolbox/Types.hpp>

#include <utility>
#include <type_traits>

namespace AIToolbox {
    /**
     * @brief This class is a simple iterator to iterate over a container with the specified ids.
     */
    template <typename IdsIterator, typename Container>
    class IndexMapIterator {
        public:
            using value_type = typename Container::value_type;

            /**
             * @brief Basic constructor for begin iterators.
             *
             * @param parent The parent iterable object holding ids and values.
             */
            IndexMapIterator(IdsIterator id, Container & items) :
                    currentId_(id), items_(items) {}

            auto& operator*()  { return items_[toContainerId()]; }
            const auto& operator*() const { return items_[toContainerId()]; }

            auto  operator->() { return &(operator*()); }
            auto  operator->() const { return &(operator*()); }

            /**
             * @brief This function returns the equivalent item id of this iterator in its container.
             */
            auto toContainerId() const { return *currentId_; }

            auto operator++() {
                ++currentId_;
                return *this;
            }

            bool operator==(const IndexMapIterator & other) {
                return (&(items_) == &(other.items_)) &&
                       (currentId_ == other.currentId_);
            }
            bool operator!=(const IndexMapIterator & other) { return !(*this == other); }

        private:
            IdsIterator currentId_;
            Container & items_;
    };

    /**
     * @brief This class is an iterable construct on a list of ids on a given container.
     *
     * This class allows to iterate over a given subset of ids on the input
     * container as if they were laid out continuously in a single object.
     *
     * Both ids and items must be stored in containers accessible via
     * square-brakets and ids.
     *
     * By default the iterable will copy the input ids and own them. If this is not
     * desirable (maybe one wants to change the ids over time without being forced
     * to copy them multiple times), the class accepts a pointer to an ids
     * container, and it will automatically store a single reference to it, rather
     * than doing a copy.
     *
     * @tparam IdsContainer The type of the input ids in the constructor.
     * @tparam Container The type of the container to be iterated on.
     */
    template <typename IdsContainer, typename Container>
    class IndexMap {
        public:
            /**
             * @brief The type used to contain the ids in the iterable.
             *
             * This is a constant copy of the input ids if we own them, and
             * otherwise a const reference if we don't (and thus they can change).
             */
            static constexpr bool OwnsIds = !std::is_pointer_v<IdsContainer>;
            using IdsStorage  = std::conditional_t<OwnsIds,
                                             const IdsContainer,
                                             const std::remove_pointer_t<IdsContainer> &
                                >;

            using value_type     = typename Container::value_type;

            using iterator       = IndexMapIterator<decltype(std::declval<IdsStorage>().begin()),  Container>;
            using const_iterator = IndexMapIterator<decltype(std::declval<IdsStorage>().cbegin()), const Container>;

            /**
             * @brief Basic constructor for owning iterable.
             *
             * This constructor stores a copy of all the ids and a reference to the
             * container over which to iterate.
             *
             * This class and its iterators do *NOT* perform any bound checking on
             * the size of the container and the input ids, neither at construction
             * nor during operation.
             *
             * This class and its iterators *WILL* be invalidated if the item
             * container is destroyed.
             *
             * @param ids The ids to iterate over.
             * @param items The items container.
             */
            template <bool Tmp = OwnsIds, typename std::enable_if_t<Tmp, int> = 0>
            IndexMap(IdsContainer ids, Container & items) : ids_(std::move(ids)), items_(items) {}

            /**
             * @brief Basic constructor for non-owning iterable.
             *
             * This constructor stores the pointer to the ids and items over which
             * to iterate.
             *
             * This class and its iterators do *NOT* perform any bound checking on
             * the size of the container and the input ids, neither at construction
             * nor during operation.
             *
             * This class and its iterators *WILL* be invalidated if the ids
             * container or the item container are destroyed.
             *
             * If the ids change, all previously generated iterators are invalidated.
             *
             * @param ids The ids to iterate over.
             * @param items The items container.
             */
            template <bool Tmp = OwnsIds, typename std::enable_if_t<!Tmp, int> = 0>
            IndexMap(IdsContainer ids, Container & items) : ids_(*ids), items_(items) {}

            /**
             * @brief This function returns an iterator to the beginning of this filtered range.
             */
            auto begin() { return iterator(ids_.begin(), items_); }

            /**
             * @brief This function returns a const_iterator to the beginning of this filtered range.
             */
            auto begin() const { return cbegin(); }

            /**
             * @brief This function returns a const_iterator to the beginning of this filtered range.
             */
            auto cbegin() const { return const_iterator(ids_.cbegin(), items_); }

            /**
             * @brief This function returns an iterator to the end of this filtered range.
             */
            auto end() { return iterator(ids_.end(), items_); };

            /**
             * @brief This function returns a const_iterator to the end of this filtered range.
             */
            auto end() const { return cend(); }

            /**
             * @brief This function returns a const_iterator to the end of this filtered range.
             */
            auto cend() const { return const_iterator(ids_.cend(), items_); }

            /**
             * @brief This function returns the size of the range covered.
             */
            auto size() const { return ids_.size(); }

        private:
            friend iterator;
            friend const_iterator;

            // Const reference if non-owning, const value otherwise.
            IdsStorage ids_;
            Container & items_;
    };
}
