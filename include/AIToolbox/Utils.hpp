#ifndef AI_TOOLBOX_UTILS_HEADER_FILE
#define AI_TOOLBOX_UTILS_HEADER_FILE

#include <stddef.h>
#include <cmath>

namespace AIToolbox {
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
    void copyTable3D(const T & in, U & out, size_t d1, size_t d2, size_t d3) {
        for ( size_t i = 0; i < d1; ++i )
            for ( size_t j = 0; j < d2; ++j )
                for ( size_t x = 0; x < d3; ++x )
                    out[i][j][x] = in[i][j][x];
    }

    /**
     * @brief This function checks whether the supplied table is a correct probability table.
     *
     * This function verifies basic probability conditions on the
     * supplied container. The sum of all rows along the second
     * dimension must be 1.
     *
     * The container needs to support data access through
     * operator[]. In addition, the dimensions of the
     * container must match the ones provided as arguments
     * (for three dimensions: d1,d2,d3).
     *
     * This is important, as this function DOES NOT perform
     * any size checks on the external containers.
     *
     * Internal values of the container will be converted to double,
     * so that convertion must be possible.
     *
     * @tparam T The external transition container type.
     * @param in The external transitions container.
     * @param d1 The size along the first dimension of the supplied container.
     * @param d2 The size along the second dimension of the supplied container.
     * @param d3 The size along the third dimension of the supplied container.
     *
     * @return True if the container statisfies probability constraints,
     *         and false otherwise.
     */
    template <typename T>
    bool transitionCheck(const T & in, size_t d1, size_t d2, size_t d3) {
        for ( size_t i = 0; i < d1; ++i ) {
            for ( size_t x = 0; x < d3; ++x ) {
                double p = 0.0;
                for ( size_t j = 0; j < d2; ++j ) {
                    p += in[i][j][x];
                }
                if ( fabs(p - 1.0) > 0.000001 ) return false;
            }
        }
        return true;
    }
}

#endif
