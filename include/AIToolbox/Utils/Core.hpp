#ifndef AI_TOOLBOX_UTILS_CORE_HEADER_FILE
#define AI_TOOLBOX_UTILS_CORE_HEADER_FILE

#include <cmath>
#include <limits>

#include <AIToolbox/Types.hpp>

#include <boost/functional/hash.hpp>

namespace AIToolbox {
    constexpr auto equalToleranceSmall = 0.000001;
    constexpr auto equalToleranceGeneral = 0.00000000001;
    /**
     * @brief Copies a 3d container into another 3d container.
     *
     * The containers needs to support data access through
     * operator[]. In addition, the dimensions of the
     * containers must match the ones specified.
     *
     * This is important, as this function DOES NOT perform
     * any size checks on the containers.
     *
     * @tparam T Type of the input container.
     * @tparam U Type of the output container.
     * @param in Input container.
     * @param out Output container.
     * @param d1 First dimension of the containers.
     * @param d2 Second dimension of the containers.
     * @param d3 Third dimension of the containers.
     */
    template <typename T, typename U>
    void copyTable3D(const T & in, U & out, const size_t d1, const size_t d2, const size_t d3) {
        for ( size_t i = 0; i < d1; ++i )
            for ( size_t j = 0; j < d2; ++j )
                for ( size_t x = 0; x < d3; ++x )
                    out[i][j][x] = in[i][j][x];
    }

    /**
     * @brief This function checks if two doubles near [0,1] are reasonably equal.
     *
     * If the numbers are not near [0,1], the result is not guaranteed to be
     * what may be expected. The order of the parameters is not important.
     *
     * @param a The first number to compare.
     * @param b The second number to compare.
     *
     * @return True if the two numbers are close enough, false otherwise.
     */
    inline bool checkEqualSmall(const double a, const double b) {
        return ( std::fabs(a - b) <= equalToleranceSmall );
    }

    /**
     * @brief This function checks if two doubles near [0,1] are reasonably different.
     *
     * If the numbers are not near [0,1], the result is not guaranteed to be
     * what may be expected. The order of the parameters is not important.
     *
     * @param a The first number to compare.
     * @param b The second number to compare.
     *
     * @return True if the two numbers are far away enough, false otherwise.
     */
    inline bool checkDifferentSmall(const double a, const double b) {
        return !checkEqualSmall(a,b);
    }

    /**
     * @brief This function checks if two doubles are reasonably equal.
     *
     * The order of the parameters is not important.
     *
     * @param a The first number to compare.
     * @param b The second number to compare.
     *
     * @return True if the two numbers are close enough, false otherwise.
     */
    inline bool checkEqualGeneral(const double a, const double b) {
        if ( checkEqualSmall(a,b) ) return true;
        return ( std::fabs(a - b) <= std::min(std::fabs(a), std::fabs(b)) * equalToleranceGeneral );
    }

    /**
     * @brief This function checks if two doubles are reasonably different.
     *
     * The order of the parameters is not important.
     *
     * @param a The first number to compare.
     * @param b The second number to compare.
     *
     * @return True if the two numbers are far away enough, false otherwise.
     */
    inline bool checkDifferentGeneral(const double a, const double b) {
        return !checkEqualGeneral(a,b);
    }

    /**
     * @brief This function compares two general vectors of equal size lexicographically.
     *
     * Note that veccmp reports equality only if the elements are all exactly
     * the same. You should not use this function to compare floating point
     * numbers unless you know what you are doing.
     *
     * @param lhs The left hand size of the comparison.
     * @param rhs The right hand size of the comparison.
     *
     * @return 1 if the lhs is greater than the rhs, 0 if they are equal, -1 otherwise.
     */
    template <typename V>
    int veccmp(const V & lhs, const V & rhs) {
        assert(lhs.size() == rhs.size());
        for (decltype(lhs.size()) i = 0; i < lhs.size(); ++i) {
            if (lhs[i] > rhs[i]) return 1;
            if (lhs[i] < rhs[i]) return -1;
        }
        return 0;
    }

    /**
     * @brief This function compares two general vectors of equal size lexicographically.
     *
     * Note that veccmpSmall considers two elements equal using the
     * checkEqualSmall function.
     *
     * @param lhs The left hand size of the comparison.
     * @param rhs The right hand size of the comparison.
     *
     * @return 1 if the lhs is greater than the rhs, 0 if they are equal, -1 otherwise.
     */
    template <typename V>
    int veccmpSmall(const V & lhs, const V & rhs) {
        assert(lhs.size() == rhs.size());
        for (decltype(lhs.size()) i = 0; i < lhs.size(); ++i) {
            if (checkEqualSmall(lhs[i], rhs[i])) continue;
            return lhs[i] > rhs[i] ? 1 : -1;
        }
        return 0;
    }

    /**
     * @brief This function compares two general vectors of equal size lexicographically.
     *
     * Note that veccmpSmall considers two elements equal using the
     * checkEqualGeneral function.
     *
     * @param lhs The left hand size of the comparison.
     * @param rhs The right hand size of the comparison.
     *
     * @return 1 if the lhs is greater than the rhs, 0 if they are equal, -1 otherwise.
     */
    template <typename V>
    int veccmpGeneral(const V & lhs, const V & rhs) {
        assert(lhs.size() == rhs.size());
        for (decltype(lhs.size()) i = 0; i < lhs.size(); ++i) {
            if (checkEqualGeneral(lhs[i], rhs[i])) continue;
            return lhs[i] > rhs[i] ? 1 : -1;
        }
        return 0;
    }

    /**
     * @brief This function returns whether a sorted range contains a given element, via sequential scan.
     *
     * The idea behind this function is that for small sorted vectors it is
     * faster to do a sequential scan rather than employing the heavy handed
     * technique of binary search.
     *
     * @tparam V The type of the vector to be scanned.
     * @param v The vector to be scanned.
     * @param elem The element to be looked for in the vector.
     *
     * @return True if the vector contains that element, false otherwise.
     */
    template <typename V>
    bool sequential_sorted_contains(const V & v, decltype(v[0]) elem) {
        // Maybe this could be done with iterators?
        for (const auto & e : v) {
            if (e < elem) continue;
            if (e == elem) return true;
            return false;
        }
        return false;
    }
}

namespace Eigen {
    /**
     * @brief This function enables hashing of Vectors with boost::hash.
     */
    inline size_t hash_value(const AIToolbox::Vector & v) {
        return boost::hash_range(v.data(), v.data() + v.size());
    }
}

#endif
