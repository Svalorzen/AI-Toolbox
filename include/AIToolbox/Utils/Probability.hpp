#ifndef AI_TOOLBOX_UTILS_PROBABILITY_HEADER_FILE
#define AI_TOOLBOX_UTILS_PROBABILITY_HEADER_FILE

#include <random>
#include <algorithm>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox {
    static std::uniform_real_distribution<double> probabilityDistribution(0.0, 1.0);

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
        double p = probabilityDistribution(generator);

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
        double p = probabilityDistribution(generator);

        for ( SparseMatrix2D::ConstRowXpr::InnerIterator i(in, 0); ; ++i ) {
            if ( i.value() > p ) return i.col();
            p -= i.value();
        }
        return d-1;
    }

    /**
     * @brief This function samples from a Beta distribution.
     *
     * The Beta distribution can be useful as it is the conjugate prior of the
     * Bernoulli and Binomial distributions (and others).
     *
     * As C++ does not yet have a Beta distribution in the standard, we emulate
     * the sampling using two gamma distributions.
     *
     * @tparam G The type of the generator used.
     * @param a The 'a' shape parameter of the Beta distribution to sample.
     * @param b The 'b' shape parameter of the Beta distribution to sample.
     * @param generator A random number generator.
     *
     * @return The sampled number.
     */
    template <typename G>
    double sampleBetaDistribution(double a, double b, G & generator) {
        std::gamma_distribution<double> dista(a, 1.0);
        std::gamma_distribution<double> distb(b, 1.0);
        const auto X = dista(generator);
        const auto Y = distb(generator);
        return X / (X + Y);
    }

    /**
     * @brief This function samples from the input Dirichlet distribution.
     *
     * The input parameters's type must support size() and square bracket
     * access.
     *
     * @param params The parameters of the Dirichlet distribution.
     * @param generator The random generator to sample from.
     *
     * @return A ProbabilityVector containing the sampled Dirichlet.
     */
    template <typename TIn, typename G>
    ProbabilityVector sampleDirichletDistribution(const TIn & params, G & generator) {
        ProbabilityVector retval(params.size());

        sampleDirichletDistribution(params, generator, retval);

        return retval;
    }

    /**
     * @brief This function samples from the input Dirichlet distribution inline.
     *
     * The input parameters's type must support size() and square bracket
     * access. The output parameter's type must be a dense Eigen type (Vector,
     * row, etc).
     *
     * @param params The parameters of the Dirichlet distribution.
     * @param generator The random generator to sample from.
     * @param out The output container.
     */
    template <typename TIn, typename TOut, typename G>
    void sampleDirichletDistribution(const TIn & params, G & generator, TOut && out) {
        assert(params.size() == out.size());

        double sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(params.size()); ++i) {
            std::gamma_distribution<double> dist(params[i], 1.0);
            out[i] = dist(generator);
            sum += out[i];
        }
        out /= sum;
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
            bData[s] = probabilityDistribution(generator);

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
     * @brief This function returns the entropy of the input ProbabilityVector.
     *
     * @param v The input ProbabilityVector.
     *
     * @return The entropy of the input.
     */
    inline double getEntropy(const ProbabilityVector & v) {
        return (v.array() * v.array().log()).sum();
    }

    /**
     * @brief This function returns the entropy of the input ProbabilityVector computed using log2.
     *
     * @param v The input ProbabilityVector.
     *
     * @return The entropy of the input in base 2.
     */
    inline double getEntropyBase2(const ProbabilityVector & v) {
        double entropy = 0.0;
        for (auto i = 0; i < v.size(); ++i)
            entropy += v[i] * std::log2(v[i]);
        return entropy;
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
    ProbabilityVector projectToProbability(const Vector & v);

    /**
     * @brief This class represents the Alias sampling method.
     *
     * This is an O(1) way to sample from a fixed distribution. Construction
     * takes O(N).
     *
     * The class stores two vectors of size N, and converts the input
     * probability distribution into a set of N weighted coins, each of which
     * represents a choice between two particular numbers.
     *
     * When sampled, the class simply decides which coin to use, and it rolls
     * it. This is much faster than the sampleProbability method, which is
     * O(N), as it needs to iterate over the input probability vector.
     *
     * This is the preferred method of sampling for distributions that
     * generally do not change (as if the distribution changes, the instance of
     * VoseAliasSampler must be rebuilt).
     */
    class VoseAliasSampler {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param p The probability distribution to sample from.
             */
            VoseAliasSampler(const ProbabilityVector & p);

            /**
             * @brief This function samples a number that follows the distribution of the class.
             *
             * @param generator A random number generator.
             *
             * @return A number between 0 and the size of the original ProbabilityVector.
             */
            template <typename G>
            size_t sampleProbability(G & generator) const {
                const auto x = sampleDistribution_(generator);
                const int i = x;
                const auto y = x - i;

                if (y < prob_[i]) return i;
                return alias_[i];
            }

        private:
            Vector prob_;
            std::vector<size_t> alias_;
            mutable std::uniform_real_distribution<double> sampleDistribution_;
    };
}

#endif
