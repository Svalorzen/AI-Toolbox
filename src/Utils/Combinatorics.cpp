#include <AIToolbox/Utils/Combinatorics.hpp>

namespace AIToolbox {
    unsigned nChooseK(const unsigned n, unsigned k) {
        if (k > n) return 0;
        if (k * 2 > n) k = n-k;
        if (k == 0) return 1;

        auto result = n;
        for (unsigned i = 2; i <= k; ++i) {
            result *= (n-i+1);
            result /= i;
        }
        return result;
    }

    unsigned starsBars(const unsigned stars, const unsigned bars) {
        return nChooseK(stars + bars, bars);
    }

    unsigned ballsBins(const unsigned balls, const unsigned bins) {
        return starsBars(balls, bins - 1);
    }

    unsigned nonZeroStarsBars(const unsigned stars, const unsigned bars) {
        return nChooseK(stars - 1, bars);
    }

    unsigned nonZeroBallsBins(const unsigned balls, const unsigned bins) {
        return nonZeroStarsBars(balls, bins - 1);
    }

    // SubsetEnumerator

    SubsetEnumerator::SubsetEnumerator(const size_t elementsN, const size_t limit) :
            limit_(limit), ids_(elementsN)
    {
        reset();
    }

    size_t SubsetEnumerator::advance() {
        auto current = ids_.size() - 1;
        auto limit = limit_ - 1;
        while (current && ids_[current] == limit) --current, --limit;

        auto lowest = current; // Last element we need to change.
        limit = ++ids_[current];

        while (++current != ids_.size()) ids_[current] = ++limit;

        return lowest;
    }

    bool SubsetEnumerator::isValid() const {
        return ids_.back() < limit_;
    }

    void SubsetEnumerator::reset() {
        std::iota(std::begin(ids_), std::end(ids_), 0);
    }

    unsigned SubsetEnumerator::subsetsSize() const { return nChooseK(limit_, ids_.size()); }
    size_t SubsetEnumerator::size() const { return ids_.size(); }

    const SubsetEnumerator::IdsStorage& SubsetEnumerator::operator*() const {
        return ids_;
    }

    const SubsetEnumerator::IdsStorage* SubsetEnumerator::operator->() const {
        return &ids_;
    }
}
