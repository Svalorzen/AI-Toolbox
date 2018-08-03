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
}
