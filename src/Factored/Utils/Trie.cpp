#include <AIToolbox/Factored/Utils/Trie.hpp>

#include <boost/range/adaptor/reversed.hpp>

namespace AIToolbox::Factored {
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
            if (beginNamedFilter == endNamedFilter) ++beginUnnamedFilter;
            else if (beginUnnamedFilter == endUnnamedFilter) ++beginNamedFilter;
            else *beginNamedFilter < *beginUnnamedFilter ? ++beginNamedFilter : ++beginUnnamedFilter;
        }

        bool Filter::isValid() const {
            return beginUnnamedFilter < endUnnamedFilter || beginNamedFilter < endNamedFilter;
        }

        size_t Filter::getMin() const {
            assert( isValid() );
            if (beginNamedFilter == endNamedFilter) return *beginUnnamedFilter;
            if (beginUnnamedFilter == endUnnamedFilter) return *beginNamedFilter;
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

    Trie::Trie(Factors f) : F(std::move(f)), counter_(0) {
        if ( F.size() < 2 ) throw std::invalid_argument("Factors must have at least 2 elements!");

        ids_.resize(F.size());
        for ( size_t i = 0; i < F.size(); ++i )
            ids_[i].resize(F[i]+1); // One vector per value + 1 for unnamed
    }

    Factors Trie::getF() const {
        return F;
    }

    void Trie::reserve(size_t size) {
        for (auto && v : ids_)
            v.reserve(size);
    }

    size_t Trie::size() const {
        // Pick shortest set of vectors to merge.
        const auto & toCount = ids_[std::min_element(std::begin(F), std::end(F)) - std::begin(F)];
        // Match all by merging all ids in all vectors.
        size_t retval = toCount[0].size();;
        for (size_t i = 1; i < ids_[0].size(); ++i)
            retval += toCount[i].size();

        return retval;
    }

    size_t Trie::insert(const PartialFactors & pf) {
        // We count all factors.
        size_t factor = 0;

        for ( size_t i = 0; i < pf.first.size(); ++factor ) {
            // If this factor is less than the one we are currently looking
            // at, then it is not mentioned. Thus we add it to the end of the
            // unnamed ids list for this factor (last vector).
            if (factor < pf.first[i]) {
                ids_[factor].back().push_back(counter_);
                continue;
            }
            // Otherwise we get the value, and increase i since we are done
            // with this element of the input.
            const size_t value = pf.second[i++];
            // We append the element to the ids list for this factor in the
            // vector corresponding the input value for this factor.
            ids_[factor][value].push_back(counter_);
        }
        // If any other factor is still left unmentioned, add the input at
        // the end of their unnamed range.
        for ( ; factor < F.size(); ++factor )
            ids_[factor].back().push_back(counter_);
        return counter_++;
    }

    std::vector<size_t> Trie::filter(const Factors & f, size_t offset) const {
        if (!f.size())
            return getAllIds();

        std::vector<Filter> filters;
        filters.reserve(f.size());
        // For each factor
        for ( size_t i = offset; i < f.size() + offset; ++i ) {
            auto id = i - offset;
            // Create a filter, made by two ranges: the first is the one
            // containing all ids that specified this exact key-value pair, the
            // other for the ones that did not specify anything.
            Filter filter(
                std::begin(ids_[i][f[id]]),
                std::end  (ids_[i][f[id]]),
                std::begin(ids_[i].back()),
                std::end  (ids_[i].back())
            );
            if (!filter.isValid())
                return {};
            filters.insert(std::upper_bound(std::begin(filters), std::end(filters), filter), filter);
        }
        return applyFilters(filters);
    }

    std::vector<size_t> Trie::filter(const PartialFactors & pf) const {
        if (!pf.first.size())
            return getAllIds();

        std::vector<Filter> filters;
        filters.reserve(pf.first.size());
        // For each factor
        for ( size_t i = 0; i < pf.first.size(); ++i ) {
            auto key = pf.first[i];
            auto value = pf.second[i];
            // Create a filter, made by two ranges: the first is the one
            // containing all ids that specified this exact key-value pair, the
            // other for the ones that did not specify anything.
            Filter filter(
                std::begin(ids_[key][value]),
                std::end  (ids_[key][value]),
                std::begin(ids_[key].back()),
                std::end  (ids_[key].back())
            );
            if (!filter.isValid())
                return {};
            filters.insert(std::upper_bound(std::begin(filters), std::end(filters), filter), filter);
        }
        return applyFilters(filters);
    }

    std::vector<size_t> Trie::refine(const std::vector<size_t> & ids, const PartialFactors & pf) const {
        if (!ids.size() || !pf.first.size()) {
            // If nothing to match, match all
            return ids;
        }
        std::vector<Filter> filters;
        filters.reserve(pf.first.size() + 1);
        // ids filter; we put the ids in the "unnamed" part since it's slightly
        // more efficient like this (we do fewer checks).
        filters.emplace_back(
            std::end(ids), std::end(ids),
            std::begin(ids), std::end(ids)
        );
        // For each factor
        for ( size_t i = 0; i < pf.first.size(); ++i ) {
            auto key = pf.first[i];
            auto value = pf.second[i];
            // Create a filter, made by two ranges: the first is the one
            // containing all ids that specified this exact key-value pair, the
            // other for the ones that did not specify anything.
            Filter filter(
                std::begin(ids_[key][value]),
                std::end  (ids_[key][value]),
                std::begin(ids_[key].back()),
                std::end  (ids_[key].back())
            );
            if (!filter.isValid())
                return {};
            filters.insert(std::upper_bound(std::begin(filters), std::end(filters), filter), filter);
        }
        return applyFilters(filters);
    }

    void Trie::erase(size_t id) {
        for (auto & v : ids_) {
            // FIXME: use C++20 ranges.
            // We go in reverse here since in general it's more likely to find
            // an id in the unnamed section.
            for (auto & vv : boost::adaptors::reverse(v)) {
                auto it = std::lower_bound(std::begin(vv), std::end(vv), id);
                if (it != std::end(vv) && *it == id) {
                    vv.erase(it);
                    // We break since we found the vector which contained the id.
                    break;
                }
            }
        }
    }

    void Trie::erase(size_t id, const PartialFactors & pf) {
        size_t factor = 0;
        for ( size_t i = 0; i < pf.first.size(); ++factor ) {
            auto & v = ids_[factor];
            // If this factor is less than the one we are currently looking
            // at, then it is not mentioned. Thus we add it to the end of the
            // ids list for this factor (where the unnamed ids are).
            if (factor < pf.first[i]) {
                auto it = std::lower_bound(std::begin(v.back()), std::end(v.back()), id);
                if (it != std::end(v.back()) && *it == id)
                    v.back().erase(it);
                continue;
            }
            // Otherwise we get the value, and increase i since we are done
            // with this element of the input.
            const size_t value = pf.second[i++];
            // We add the element to the ids list for this factor in the
            // range matching the input value for this factor.
            auto it = std::lower_bound(std::begin(v[value]), std::end(v[value]), id);
            if (it != std::end(v[value]) && *it == id)
                v[value].erase(it);
        }
        // If any other factor is still left unmentioned, remove the input at
        // the end of any of them.
        for ( ; factor < F.size(); ++factor ) {
            auto & v = ids_[factor];
            auto it = std::lower_bound(std::begin(v.back()), std::end(v.back()), id);
            if (*it == id)
                v.back().erase(it);
        }
    }

    std::vector<size_t> Trie::getAllIds() const {
        // Pick shortest set of vectors to merge.
        const auto & toMerge = ids_[std::min_element(std::begin(F), std::end(F)) - std::begin(F)];
        // Reserve enough space for all the indeces we are currently store.
        size_t reserve = 0;
        for (const auto & ids : toMerge)
            reserve += ids.size();
        // Match all by merging all ids in all vectors.
        std::vector<size_t> retval(reserve);
        auto it = std::copy(std::begin(toMerge[0]), std::end(toMerge[0]), std::begin(retval));
        for (size_t i = 1; i < ids_[0].size(); ++i) {
            auto newIt = std::copy(std::begin(toMerge[i]), std::end(toMerge[i]), it);
            std::inplace_merge(std::begin(retval), it, newIt);
            it = newIt;
        }
        assert(it == std::end(retval));
        return retval;
    }

    const Factors & Trie::getFactors() const { return F; }
}
