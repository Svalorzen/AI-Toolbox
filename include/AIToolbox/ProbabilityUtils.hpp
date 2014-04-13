#ifndef AI_TOOLBOX_PROBABILITY_UTILS_HEADER_FILE
#define AI_TOOLBOX_PROBABILITY_UTILS_HEADER_FILE

#include <cstddef>
#include <cmath>
#include <limits>
#include <random>

namespace AIToolbox {
    /**
     * @brief This function checks whether the supplied vector is a correct probability vector.
     *
     * This function verifies basic probability conditions on the
     * supplied container. The sum of all elements must be 1, and
     * all elements must be >= 0 and <= 1.
     *
     * The container needs to support data access through
     * operator[]. In addition, the dimension of the
     * container must match the one provided as argument.
     *
     * This is important, as this function DOES NOT perform
     * any size checks on the external container.
     *
     * Internal values of the container will be converted to double,
     * so the conversion T to double must be possible.
     *
     * @tparam T The external transition container type.
     * @param in The external transitions container.
     * @param d The size of the supplied container.
     *
     * @return True if the container satisfies probability constraints,
     *         and false otherwise.
     */
    template <typename T>
    bool isProbability(const T & in, size_t d) {
        double p = 0.0;
        for ( size_t i = 0; i < d; ++i ) {
            double value = static_cast<double>(in[i]);
            if ( value < 0.0 || value > 1.0 ) return false;
            p += value;
        }
        if ( std::fabs(p - 1.0) > std::numeric_limits<double>::epsilon() ) 
            return false;

        return true;
    }

    /**
     * @brief This function samples an index from a probability vector.
     * 
     * This function randomly samples an index between 0 and d, given a
     * vector containing the probabilities of sampling each of the indexes.
     *
     * For performance reasons this function does not verify that the input
     * container is effectively a probability.
     *
     * The generator has to be supplied to the function, so that different
     * objects are able to maintain different generators, to reduce correlations
     * between different samples. The generator has to be compatible with
     * std::uniform_real_distribution<double>, since that is what is used
     * to obtain the random sample.
     *
     * @tparam T The type of the container vector to sample.
     * @tparam G The type of the generator used.
     * @param in The external probability container.
     * @param d The size of the supplied container.
     * @param generator The generator used to sample.
     *
     * @return An index in range [0,d-1].
     */
    template <typename T, typename G>
    size_t sampleProbability(const T& in, size_t d, G& generator) {
        static std::uniform_real_distribution<double> sampleDistribution(0.0, 1.0);
        double p = sampleDistribution(generator);

        for ( size_t i = 0; i < d; ++i ) {
            if ( in[i] > p ) return i;
            p -= in[i];
        }
        return d-1;
    }
}

#endif
