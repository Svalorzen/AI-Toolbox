#include <AIToolbox/FactoredMDP/FactoredContainer.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        namespace {
            /**
             * @brief This class is used in the Trie in order easily merge id lists.
             */
            class Filter {
                public:
                    using It = std::vector<size_t>::const_iterator;

                    /**
                     * @brief Basic constructor.
                     *
                     * This constructor holds the ranges for the named range holding all ids matching the
                     * input at a specific depth, and the correspondend range for all rules which did
                     * not have a specific value at this depth. All these ids can potentially match
                     * the input and must thus be taken in consideration.
                     */
                    Filter(It bnamed, It enamed, It bunnamed, It eunnamed);

                    /**
                     * @brief This function advances both iterators so that their pointed value is at least the input.
                     *
                     * @param value The minimum value to be matched by both iterators.
                     *
                     * @return The minimum value pointed by either iterator after the move.
                     */
                    size_t advance(size_t value);

                    /**
                     * @brief This function advances by a single step the minimum iterator.
                     *
                     * @return The minimum value pointed by either iterator after the move.
                     */
                    size_t stepAdvance();

                    /**
                     * @brief This function returns whether this filted is still valid, i.e. at least one iterator has still items to check.
                     */
                    bool isValid() const;

                    /**
                     * @brief This function returns the minimum value pointed by either iterator, if valid.
                     */
                    size_t getMin() const;

                    /**
                     * @brief This operator sorts Filters by total number of ids in its ranges.
                     */
                    bool operator<(const Filter & other) const;

                private:
                    It beginNamedFilter, endNamedFilter;
                    It beginUnnamedFilter, endUnnamedFilter;
            };

            Filter::Filter(It bnamed, It enamed, It bunnamed, It eunnamed) :
                beginNamedFilter(bnamed), endNamedFilter(enamed),
                beginUnnamedFilter(bunnamed), endUnnamedFilter(eunnamed) {}

            size_t Filter::advance(size_t value) {
                beginNamedFilter = std::lower_bound(beginNamedFilter, endNamedFilter, value);
                beginUnnamedFilter = std::lower_bound(beginUnnamedFilter, endUnnamedFilter, value);
                return getMin();
            }

            size_t Filter::stepAdvance() {
                if (beginUnnamedFilter == endUnnamedFilter) ++beginNamedFilter;
                else if (beginNamedFilter == endNamedFilter) ++beginUnnamedFilter;
                else *beginNamedFilter < *beginUnnamedFilter ? ++beginNamedFilter : ++beginUnnamedFilter;
                return getMin();
            }

            bool Filter::isValid() const {
                return beginNamedFilter < endNamedFilter || beginUnnamedFilter < endUnnamedFilter;
            }

            size_t Filter::getMin() const {
                if ( !isValid() ) return 0;
                if (beginUnnamedFilter == endUnnamedFilter) return *beginNamedFilter;
                if (beginNamedFilter == endNamedFilter) return *beginUnnamedFilter;
                return std::min(*beginNamedFilter, *beginUnnamedFilter);
            }

            bool Filter::operator<(const Filter & other) const {
                return (endNamedFilter - beginNamedFilter) + (endUnnamedFilter - beginUnnamedFilter) <
                       (other.endNamedFilter - other.beginNamedFilter) + (other.endUnnamedFilter - other.beginUnnamedFilter);
            }
        }

        Trie::Trie(Factors s) : S(s), counter_(0) {
            if ( s.size() < 2 ) throw std::invalid_argument("Factors must have at least 2 elements!");

            partials_.resize(S.size());
            for ( size_t i = 0; i < S.size(); ++i )
                partials_[i].resize(S[i], 0);

            ids_.resize(S.size());
        }

        Factors Trie::getS() const {
            return S;
        }

        void Trie::reserve(size_t size) {
            for (auto && v : ids_)
                v.reserve(size);
        }

        size_t Trie::size() const {
            return counter_;
        }

        void Trie::insert(const PartialFactors & ps) {
            size_t factor = 0;

            for ( size_t counter = 0; counter < ps.first.size(); ++factor ) {
                if ( factor < ps.first[counter]) {
                    ids_[factor].push_back(counter_);
                    continue;
                }

                size_t value = 0;
                value = ps.second[counter++];

                ids_[factor].insert(std::begin(ids_[factor]) + partials_[factor][value], counter_);

                for (; value < S[factor]; ++value)
                    ++partials_[factor][value];
            }
            for ( ; factor < S.size(); ++factor )
                ids_[factor].push_back(counter_);
            ++counter_;
        }

        std::vector<size_t> Trie::filter(const Factors & s) const {
            std::vector<size_t> matches;

            std::vector<Filter> filters;
            filters.reserve(S.size());
            // For each factor
            for ( size_t i = 0; i < S.size(); ++i ) {
                // Create a filter, made by two ranges: the first is the one
                // containing all factors that specified this exact factor, the
                // other for the ones that did not specify anything.
                Filter f(
                    std::begin(ids_[i]) + (s[i] == 0 ? 0 : partials_[i][s[i]-1]),
                    std::begin(ids_[i]) + (partials_[i][s[i]]),
                    std::begin(ids_[i]) + (partials_[i].back()),
                    std::end(ids_[i])
                );
                auto it = filters.insert(std::upper_bound(std::begin(filters), std::end(filters), f), f);
                if ( !it->isValid() )
                    return matches;
            }

            size_t lastMaxFound = 0, counter = 1;
            size_t currentMax = filters[0].getMin();

            while ( true ) {
                // If we have matched through all filters
                if ( counter == S.size() ) {
                    // Match the id
                    matches.push_back(currentMax);
                    // Advance by one the top filter, and mark the id there.
                    currentMax = filters[0].stepAdvance();
                    if ( !filters[0].isValid() ) break;
                    counter = 1;
                    lastMaxFound = 0;
                }
                // Slowly advance all other filters to the max
                auto currentId = filters[counter].advance(currentMax);
                if ( !filters[counter].isValid() ) break;

                if ( currentId > currentMax ) {
                    currentMax = currentId;
                    lastMaxFound = counter;
                    counter = 0;
                } else if ( ++counter == lastMaxFound )
                    ++counter;
            }
            return matches;
        }
    }
}
