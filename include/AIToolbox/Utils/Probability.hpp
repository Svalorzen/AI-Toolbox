#ifndef AI_TOOLBOX_UTILS_PROBABILITY_HEADER_FILE
#define AI_TOOLBOX_UTILS_PROBABILITY_HEADER_FILE

#include <cstddef>
#include <random>
#include <algorithm>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/Core.hpp>

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
    bool isProbability(const size_t d, const T & in) {
        double p = 0.0;
        for ( size_t i = 0; i < d; ++i ) {
            const double value = static_cast<double>(in[i]);
            if ( value < 0.0 ) return false;
            p += value;
        }
        if ( checkDifferentSmall(p, 1.0) )
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
    size_t sampleProbability(const size_t d, const T& in, G& generator) {
        static std::uniform_real_distribution<double> sampleDistribution(0.0, 1.0);
        double p = sampleDistribution(generator);

        for ( size_t i = 0; i < d; ++i ) {
            if ( in[i] > p ) return i;
            p -= in[i];
        }
        return d-1;
    }

    /**
     * @brief This function samples an index from a sparse probability vector.
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
     * @tparam G The type of the generator used.
     * @param in The external probability container.
     * @param d The size of the supplied container.
     * @param generator The generator used to sample.
     *
     * @return An index in range [0,d-1].
     */
    template <typename G>
    size_t sampleProbability(const size_t d, const SparseMatrix2D::ConstRowXpr& in, G& generator) {
        static std::uniform_real_distribution<double> sampleDistribution(0.0, 1.0);
        double p = sampleDistribution(generator);

        for ( SparseMatrix2D::ConstRowXpr::InnerIterator i(in, 0); ; ++i ) {
            if ( i.value() > p ) return i.col();
            p -= i.value();
        }
        return d-1;
    }

    /**
     * @brief This function generates a random probability vector.
     *
     * This function will sample uniformly from the simplex space with the
     * specified number of dimensions.
     *
     * S must be at least one or we don't guarantee any behaviour.
     *
     * @param S The number of entries of the output vector.
     * @param generator A random number generator.
     *
     * @return A new random probability vector.
     */
    template <typename G>
    ProbabilityVector makeRandomProbability(const size_t S, G & generator) {
        static std::uniform_real_distribution<double> sampleDistribution(0.0, 1.0);
        ProbabilityVector b(S);
        double * bData = b.data();
        // The way this works is that we're going to generate S-1 numbers in
        // [0,1], and sort them with together with an implied 0.0 and 1.0, for
        // a total of S+1 numbers.
        //
        // The output will be represented by the differences between each pair
        // of numbers, after sorting the original vector.
        //
        // The idea is basically to take a unit vector and cut it up into
        // random parts. The size of each part is the value of an entry of the
        // output.

        // We must set the first element to zero even if we're later
        // overwriting it. This is to avoid bugs in case the input S is one -
        // in which case we should return a vector with a single element
        // containing 1.0.
        bData[0] = 0.0;
        for ( size_t s = 0; s < S-1; ++s )
            bData[s] = sampleDistribution(generator);

        // Sort all but the implied last 1.0 which we'll add later.
        std::sort(bData, bData + S - 1);

        // For each number, keep track of what was in the vector there, and
        // transform it into the difference with its predecessor.
        double helper1 = bData[0], helper2;
        for ( size_t s = 1; s < S - 1; ++s ) {
            helper2 = bData[s];
            bData[s] -= helper1;
            helper1 = helper2;
        }
        // The final one is computed with respect to the overall sum of 1.0.
        bData[S-1] = 1.0 - helper1;

        return b;
    }

    /**
     * @brief This function checks whether two input ProbabilityVector are equal.
     *
     * This function is approximate. It assumes that the vectors are valid, so
     * they must sum up to one, and each element must be between zero and one.
     * The vector must also be of the same size.
     *
     * This function is approximate, as we're dealing with floating point.
     *
     * @param lhs The left hand side to check.
     * @param rhs The right hand side to check.
     *
     * @return Whether the two ProbabilityVectors are the same.
     */
    inline bool checkEqualProbability(const ProbabilityVector & lhs, const ProbabilityVector & rhs) {
        const auto size = lhs.size();
        for (auto i = 0; i < size; ++i)
            if (!checkEqualSmall(lhs[i], rhs[i]))
                return false;
        return true;
    }

    /**
     * @brief This function projects the input vector to a valid probability space.
     *
     * This function finds the closest valid ProbabilityVector to the input
     * vector. The distance measure used here is the sum of the absolute values
     * of the element-wise difference between the input and the output.
     *
     * When it has a choice, it tries to preserve the "shape" of the input and
     * not arbitrarily change elements around.
     *
     * @param v The vector to project to a valid probability space.
     *
     * @return The closes valid probability vector to the input.
     */
    inline ProbabilityVector projectToProbability(const Vector & v) {
        ProbabilityVector retval(v.size());

        double sum = 0.0;
        size_t count = 0;
        for (auto i = 0; i < v.size(); ++i) {
            // Negative elements are converted to zero, as that's the best we
            // can do.
            if (v[i] < 0.0) retval[i] = 0.0;
            else {
                retval[i] = 1.0;
                ++count;
                sum += v[i];
            }
        }
        if (checkEqualSmall(sum, 1.0)) return retval;
        if (checkEqualSmall(sum, 0.0)) {
            // Any solution here would do, but this seems nice.
            retval.array() += 1.0 / v.size();
        } else if (sum > 1.0) {
            // We normalize the vector.
            retval.array() *= v.array() / sum;
        } else {
            // We remove equally from all non-zero elements.
            const auto diff = (1.0 - sum) / count;
            retval.array() *= (v.array() + diff);
        }
        return retval;
    }
}

#endif
