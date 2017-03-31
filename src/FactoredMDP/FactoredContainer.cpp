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
                     */
                    void advance(size_t value);

                    /**
                     * @brief This function advances by a single step the minimum iterator.
                     */
                    void stepAdvance();

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

            void Filter::advance(size_t value) {
                beginNamedFilter = std::lower_bound(beginNamedFilter, endNamedFilter, value);
                beginUnnamedFilter = std::lower_bound(beginUnnamedFilter, endUnnamedFilter, value);
            }

            void Filter::stepAdvance() {
                if (beginUnnamedFilter == endUnnamedFilter) ++beginNamedFilter;
                else if (beginNamedFilter == endNamedFilter) ++beginUnnamedFilter;
                else *beginNamedFilter < *beginUnnamedFilter ? ++beginNamedFilter : ++beginUnnamedFilter;
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

            /**
             * @brief This function finds all common elements held by the input filters.
             *
             * This function assumes that there is at least one filter
             * available, and that all input filters contain at least one valid
             * value. This function will advance all input filters, modifying
             * them.
             *
             * @param filters The input filters.
             *
             * @return A vector containing all common elements shared by the filters.
             */
            std::vector<size_t> applyFilters(std::vector<Filter> & filters) {
                std::vector<size_t> matches;

                if (filters.size() == 1) {
                    while (filters[0].isValid()) {
                        matches.push_back(filters[0].getMin());
                        filters[0].stepAdvance();
                    }
                    return matches;
                }

                size_t lastMaxFound = 0, counter = 1;
                size_t currentMax = filters[0].getMin();
                while ( true ) {
                    // If we have matched through all filters
                    if ( counter == filters.size() ) {
                        // Match the id
                        matches.push_back(currentMax);
                        // Advance by one the top filter, and mark the id there.
                        filters[0].stepAdvance();
                        if ( !filters[0].isValid() ) break;
                        currentMax = filters[0].getMin();
                        counter = 1;
                        lastMaxFound = 0;
                    }
                    // Slowly advance all other filters to the max
                    filters[counter].advance(currentMax);
                    if ( !filters[counter].isValid() ) break;
                    auto currentId = filters[counter].getMin();

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

        Trie::Trie(Factors f) : F(f), counter_(0) {
            if ( F.size() < 2 ) throw std::invalid_argument("Factors must have at least 2 elements!");

            partials_.resize(F.size());
            for ( size_t i = 0; i < F.size(); ++i )
                partials_[i].resize(F[i], 0);

            ids_.resize(F.size());
        }

        Factors Trie::getF() const {
            return F;
        }

        void Trie::reserve(size_t size) {
            for (auto && v : ids_)
                v.reserve(size);
        }

        size_t Trie::size() const {
            return counter_;
        }

        void Trie::insert(const PartialFactors & ps) {
            // We count all factors.
            size_t factor = 0;

            for ( size_t i = 0; i < ps.first.size(); ++factor ) {
                // If this factor is less than the one we are currently looking
                // at, then it is not mentioned. Thus we add it to the end of the
                // ids list for this factor (where the unnamed ids are).
                if (factor < ps.first[i]) {
                    ids_[factor].push_back(counter_);
                    continue;
                }
                // Otherwise we get the value, and increase i since we are done
                // with this element of the input.
                size_t value = ps.second[i++];
                // We add the element to the ids list for this factor in the
                // range matching the input value for this factor.
                ids_[factor].insert(std::begin(ids_[factor]) + partials_[factor][value], counter_);
                // We increase all partials from this point forward.
                for (; value < F[factor]; ++value)
                    ++partials_[factor][value];
            }
            // If any other factor is still left unmentioned, add the input at
            // the end of any of them.
            for ( ; factor < F.size(); ++factor )
                ids_[factor].push_back(counter_);
            ++counter_;
        }

        std::vector<size_t> Trie::filter(const Factors & f, size_t offset) const {
            if (!f.size()) {
                // If nothing to match, match all
                std::vector<size_t> retval(counter_);
                std::iota(std::begin(retval), std::end(retval), 0);
                return retval;
            }
            std::vector<Filter> filters;
            filters.reserve(f.size());
            // For each factor
            for ( size_t i = offset; i < f.size() + offset; ++i ) {
                auto id = i - offset;
                // Create a filter, made by two ranges: the first is the one
                // containing all factors that specified this exact factor, the
                // other for the ones that did not specify anything.
                Filter filter(
                    std::begin(ids_[i]) + (f[id] == 0 ? 0 : partials_[i][f[id]-1]),
                    std::begin(ids_[i]) + (partials_[i][f[id]]),
                    std::begin(ids_[i]) + (partials_[i].back()),
                    std::end(ids_[i])
                );
                if (!filter.isValid())
                    return {};
                filters.insert(std::upper_bound(std::begin(filters), std::end(filters), filter), filter);
            }
            return applyFilters(filters);
        }

        std::vector<size_t> Trie::filter(const PartialFactors & pf) const {
            if (!pf.first.size()) {
                // If nothing to match, match all
                std::vector<size_t> retval(counter_);
                std::iota(std::begin(retval), std::end(retval), 0);
                return retval;
            }
            std::vector<Filter> filters;
            filters.reserve(pf.first.size());
            // For each factor
            for ( size_t i = 0; i < pf.first.size(); ++i ) {
                auto factor = pf.first[i];
                auto value = pf.second[i];
                // Create a filter, made by two ranges: the first is the one
                // containing all factors that specified this exact factor, the
                // other for the ones that did not specify anything.
                Filter filter(
                    std::begin(ids_[factor]) + (value == 0 ? 0 : partials_[factor][value-1]),
                    std::begin(ids_[factor]) + (partials_[factor][value]),
                    std::begin(ids_[factor]) + (partials_[factor].back()),
                    std::end(ids_[factor])
                );
                if (!filter.isValid())
                    return {};
                filters.insert(std::upper_bound(std::begin(filters), std::end(filters), filter), filter);
            }
            return applyFilters(filters);
        }
    }
}
