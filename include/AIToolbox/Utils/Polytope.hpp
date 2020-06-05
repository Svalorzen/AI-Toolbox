#ifndef AI_TOOLBOX_UTILS_POLYTOPE_HEADER_FILE
#define AI_TOOLBOX_UTILS_POLYTOPE_HEADER_FILE

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/TypeTraits.hpp>
#include <AIToolbox/Utils/Combinatorics.hpp>
#include <Eigen/Dense>
#include <array>

#include <AIToolbox/Utils/LP.hpp>

namespace AIToolbox {
    /**
     * @brief Defines a plane in a simplex where each value is the height at that corner.
     */
    using Hyperplane = Vector;

    /**
     * @brief Defines a point inside a simplex. Coordinates sum to 1.
     */
    using Point = ProbabilityVector;

    /**
     * @brief A surface within a simplex defined by points and their height. Should not contain the corners.
     */
    using PointSurface = std::pair<std::vector<Point>, std::vector<double>>;

    /**
     * @brief A compact set of (probably |A|) hyperplanes, one per column (probably |S| rows). This is generally used with PointSurface; otherwise we use a vector<Hyperplane>.
     */
    using CompactHyperplanes = Matrix2D;

    /**
     * @brief This function checks whether an Hyperplane dominates another.
     *
     * @param lhs The Hyperplane that should dominate.
     * @param rhs The Hyperplane that should be dominated.
     *
     * @return Whether the left hand side dominates the right hand side.
     */
    inline bool dominates(const Hyperplane & lhs, const Hyperplane & rhs) {
        return (lhs.array() - rhs.array() >= -equalToleranceSmall).minCoeff() ||
               (lhs.array() - rhs.array() >= -lhs.array().min(rhs.array()) * equalToleranceGeneral).minCoeff();
    };

    /**
     * @brief This function returns an iterator pointing to the best Hyperplane for the specified point.
     *
     * Given a list of hyperplanes as a surface, this function returns the
     * hyperplane which provides the highest value at the specified point.
     *
     * @param p The point where we need to check the value
     * @param begin The start of the range to look in.
     * @param end The end of the range to look in (excluded).
     * @param value A pointer to double, which gets set to the value of the given point with the found Hyperplane.
     * @param p A projection function to call on the iterators (defaults to identity).
     *
     * @return An iterator pointing to the best choice in range.
     */
    template <typename Iterator, typename P = identity>
    Iterator findBestAtPoint(const Point & point, Iterator begin, Iterator end, double * value = nullptr, P p = P{}) {
        auto bestMatch = begin;
        double bestValue = point.dot(std::invoke(p, *bestMatch));

        while ( (++begin) < end ) {
            const double currValue = point.dot(std::invoke(p, *begin));
            if ( currValue > bestValue || ( currValue == bestValue && veccmp(std::invoke(p, *begin), std::invoke(p, *bestMatch)) > 0 ) ) {
                bestMatch = begin;
                bestValue = currValue;
            }
        }
        if ( value ) *value = bestValue;
        return bestMatch;
    }

    /**
     * @brief This function returns an iterator pointing to the best Hyperplane for the specified corner of the simplex space.
     *
     * This function is slightly more efficient than findBestAtPoint for a
     * simplex corner.
     *
     * @param corner The corner of the simplex space we are checking.
     * @param begin The start of the range to look in.
     * @param end The end of the range to look in (excluded).
     * @param p A projection function to call on the iterators (defaults to identity).
     *
     * @return An iterator pointing to the best choice in range.
     */
    template <typename Iterator, typename P = identity>
    Iterator findBestAtSimplexCorner(const size_t corner, Iterator begin, Iterator end, double * value = nullptr, P p = P{}) {
        auto bestMatch = begin;
        double bestValue = std::invoke(p, *bestMatch)[corner];

        while ( (++begin) < end ) {
            const double currValue = std::invoke(p, *begin)[corner];
            if ( currValue > bestValue || ( currValue == bestValue && veccmp(std::invoke(p, *begin), std::invoke(p, *bestMatch)) > 0 ) ) {
                bestMatch = begin;
                bestValue = currValue;
            }
        }
        if ( value ) *value = bestValue;
        return bestMatch;
    }

    /**
     * @brief This function returns, if it exists, an iterator to the highest Hyperplane that delta-dominates the input one.
     *
     * Delta-domination refers to a concept introduced in the SARSOP paper. It
     * means that a Hyperplane dominates another in a neighborhood of a given
     * Point p. This is in contrast to either simply being higher at that
     * point, or dominating the other plane across the whole simplex space.
     *
     * The returned entry of this function depends on the ordering of the range,
     * as more than one Hyperplane may delta-dominate the input, but they may
     * not delta-dominate each other.
     *
     * Thus, we iterate the input range once, and we check iteratively if an
     * entry delta-dominates the currently found best Hyperplane.
     *
     * Note that an entry is not guaranteed to exist; in that case we return
     * the end of the input range.
     *
     * @param point The Point where to check for delta-domination.
     * @param plane The Hyperplane that needs to be delta-dominated.
     * @param delta The delta value to use to validate delta-domination, i.e. the size of the neighborhood to check.
     * @param begin The start of the range to check.
     * @param end The end of the range to check.
     * @param p A projection function to call on the iterators (defaults to identity).
     *
     * @return An iterator to the highest dominating entry, or if none is found, the end of the range.
     */
    template <typename Iterator, typename P = identity>
    Iterator findBestDeltaDominated(const Point & point, const Hyperplane & plane, double delta, Iterator begin, Iterator end, P p = P{}) {
        auto retval = end;

        const Hyperplane * maxPlane = &plane;
        double maxVal = point.dot(*maxPlane);

        for (auto it = begin; it < end; ++it) {
            const Hyperplane * newPlane = &std::invoke(p, *it);
            const double newVal = point.dot(*newPlane);
            if (newVal > maxVal) {
                const double deltaValue = (newVal - maxVal) / (*newPlane - *maxPlane).norm();
                if (deltaValue > delta) {
                    maxVal = newVal;
                    maxPlane = newPlane;
                    retval = it;
                }
            }
        }
        return retval;
    }

    /**
     * @brief This function finds and moves the Hyperplane with the highest value for the given point at the beginning of the specified range.
     *
     * This function uses an already existing bound containing previously
     * marked useful hyperplanes. The order is 'begin'->'bound'->'end', where
     * bound may be equal to end where no previous bound exists. The found
     * hyperplane is moved between 'begin' and 'bound', but only if it was not
     * there previously.
     *
     * @param p The point where we need to check the value
     * @param begin The begin of the search range.
     * @param bound The begin of the 'useful' range.
     * @param end The range end to be checked. It is NOT included in the search.
     * @param p A projection function to call on the iterators (defaults to identity).
     *
     * @return The new bound iterator.
     */
    template <typename Iterator, typename P = identity>
    Iterator extractBestAtPoint(const Point & point, Iterator begin, Iterator bound, Iterator end, P p = P{}) {
        auto bestMatch = findBestAtPoint(point, begin, end, nullptr, p);

        if ( bestMatch >= bound )
            std::iter_swap(bestMatch, bound++);

        return bound;
    }

    /**
     * @brief This function finds and moves all best Hyperplanes in the simplex corners at the beginning of the specified range.
     *
     * What this function does is to find out which hyperplanes give the
     * highest value in the corner points. Since multiple corners may use the
     * same hyperplanes, the number of found hyperplanes may not be the same as
     * the number of corners.
     *
     * This function uses an already existing bound containing previously
     * marked useful hyperplanes. The order is 'begin'->'bound'->'end', where
     * bound may be equal to end where no previous bound exists. All found
     * hyperplanes are added between 'begin' and 'bound', but only if they were
     * not there previously.
     *
     * @param S The number of corners of the simplex.
     * @param begin The begin of the search range.
     * @param bound The begin of the 'useful' range.
     * @param end The end of the search range. It is NOT included in the search.
     * @param p A projection function to call on the iterators (defaults to identity).
     *
     * @return The new bound iterator.
     */
    template <typename Iterator, typename P = identity>
    Iterator extractBestAtSimplexCorners(const size_t S, Iterator begin, Iterator bound, Iterator end, P p = P{}) {
        if ( end == bound ) return bound;

        // For each corner
        for ( size_t s = 0; s < S; ++s ) {
            auto bestMatch = findBestAtSimplexCorner(s, begin, end, nullptr, p);

            if ( bestMatch >= bound )
                std::iter_swap(bestMatch, bound++);
        }
        return bound;
    }

    /**
     * @brief This function finds and moves all non-useful points at the end of the input range.
     *
     * This function helps remove points which do not support any hyperplane
     * and are thus not useful for improving the overall surface.
     *
     * This function moves all non-useful points at the end of the input
     * range, and returns the resulting iterator pointing to the first
     * non-useful point.
     *
     * When multiple Points support the same Hyperplane, the one with the best
     * value is returned.
     *
     * The Hyperplane range may contain elements which are not supported by any
     * of the input Points (although if they exist they may slow down the
     * function).
     *
     * @param pbegin The beginning of the Point range to check.
     * @param pend The end of the Point range to check.
     * @param begin The beginning of the Hyperplane range to check against.
     * @param end The end of the Hyperplane range to check against.
     * @param p A projection function to call on the iterators (defaults to identity).
     *
     * @return An iterator pointing to the first non-useful Point.
     */
    template <typename PIterator, typename VIterator, typename P = identity>
    PIterator extractBestUsefulPoints(PIterator pbegin, PIterator pend, VIterator begin, VIterator end, P p = P{}) {
        const auto pointsN  = std::distance(pbegin, pend);
        const auto entriesN = std::distance(begin, end);

        std::vector<std::pair<PIterator, double>> bestValues(entriesN, {pend, std::numeric_limits<double>::lowest()});
        const auto maxBound = pointsN < entriesN ? pend : pbegin + entriesN;

        // So the idea here is that we advance IT only if we found a Point
        // which supports a previously unsupported Hyperplane. This allows us
        // to avoid doing later work for compacting the points before the
        // bound.
        //
        // If instead the found Point takes into consideration an already
        // supported Hyperplane, then it either is better or not. If it's
        // better, we swap it with whatever was before. In both cases, the
        // Point to discard ends up at the end and we decrease the bound.
        auto it = pbegin;
        auto bound = pend;
        while (it < bound && it < maxBound) {
            double value;
            const auto vId = std::distance(begin, findBestAtPoint(*it, begin, end, &value, p));
            if (bestValues[vId].second < value) {
                if (bestValues[vId].first == pend) {
                    bestValues[vId] = {it++, value};
                    continue;
                } else {
                    bestValues[vId].second = value;
                    std::iter_swap(bestValues[vId].first, it);
                }
            }
            std::iter_swap(it, --bound);
        }
        if (it == bound) return it;

        // If all Hyperplanes have been supported by at least one Point, then
        // we can finish up the rest with less swaps and checks. Here we only
        // swap with the best if needed, otherwise we don't have to do
        // anything.
        //
        // This is because we can return one Point per Hyperplane at the most,
        // so if we're here the bound is not going to move anyway.
        while (it < bound) {
            double value;
            const auto vId = std::distance(begin, findBestAtPoint(*it, begin, end, &value, p));
            if (bestValues[vId].second < value) {
                bestValues[vId].second = value;
                std::iter_swap(bestValues[vId].first, it);
            }
            ++it;
        }
        return maxBound;
    }

    /**
     * @brief This function implements a naive vertex enumeration algorithm.
     *
     * This function goes through every subset of planes of size S, and finds
     * all vertices it can. In particular, it goes through the first list one
     * element at a time, and joins it with S-1 elements from the second list.
     *
     * Even more precisely, we take >= 1 elements from the second list. The
     * remaining elements (so that in total we still use S-1) are simply the
     * simplex boundaries, which allows us to find the corners located there.
     *
     * This method may find duplicate vertices (it does not bother to prune
     * them), as a vertex can be in the convergence of more than S planes.
     *
     * The advantage is that we do not need any linear programming, and simple
     * matrix decomposition techniques suffice.
     *
     * We do NOT return simplex corners.
     *
     * Warning: this function will return wrong vertices if the first set contains
     * the same vectors in the second set!
     *
     * Warning: the values of each vertex depends on the planes it has been
     * found of, and thus may *not* be the true value if considering all planes
     * at the same time! In other words, the same vertex may be found multiple
     * times with different values!
     *
     * This function works on ranges of Vectors.
     *
     * @param beginNew The beginning of the range of the planes to find vertices for.
     * @param endNew The end of the range of the planes to find vertices for.
     * @param alphasBegin The beginning of the range of all other planes.
     * @param alphasEnd The end of the range of all other planes.
     * @param p1 A projection function to call on the iterators of the first range (defaults to identity).
     * @param p2 A projection function to call on the iterators of the second range (defaults to identity).
     *
     * @return A non-unique list of all the vertices found.
     */
    template <typename NewIt, typename OldIt, typename P1 = identity, typename P2 = identity>
    PointSurface findVerticesNaive(NewIt beginNew, NewIt endNew, OldIt alphasBegin, OldIt alphasEnd, P1 p1 = P1{}, P2 p2 = P2{}) {
        PointSurface vertices;

        const size_t alphasSize = std::distance(alphasBegin, alphasEnd);
        if (alphasSize == 0) return vertices;
        const size_t S = std::invoke(p2, *alphasBegin).size();

        // This enumerator allows us to compute all possible subsets of S-1
        // elements. We use it on both the alphas, and the boundaries, thus the
        // number of elements we iterate over is alphasSize + S.
        SubsetEnumerator enumerator(S - 1, 0ul, alphasSize + S);

        // This is the matrix on the left side of Ax = b (where A is m)
        Matrix2D m(S + 1, S + 1);
        m.row(0)[S] = -1; // First row is always a vector

        Vector boundary(S+1);
        boundary[S] = 0.0; // The boundary doesn't care about the value

        // This is the vector on the right side of Ax = b
        Vector b(S+1); b.setZero();

        Vector result(S+1);

        // Common matrix/vector setups

        for (auto newVIt = beginNew; newVIt != endNew; ++newVIt) {
            m.row(0).head(S) = std::invoke(p1, *newVIt);

            enumerator.reset();

            // Get subset of planes, find corner with LU
            size_t last = 0;
            while (enumerator.isValid()) {
                // Reset boundaries to care about all dimensions
                boundary.head(S).fill(1.0);
                size_t counter = last + 1;
                // Note that we start from last to avoid re-copying vectors
                // that are already in the matrix in their correct place.
                for (auto i = last; i < enumerator->size(); ++i) {
                    // For each value in the enumerator, if it is less than
                    // alphasSize it is referring to an alphaVector we need to
                    // take into account.
                    const auto index = (*enumerator)[i];
                    if (index < alphasSize) {
                        // Copy the right vector in the matrix.
                        m.row(counter).head(S) = std::invoke(p2, *std::next(alphasBegin, index));
                        m.row(counter)[S] = -1;
                        ++counter;
                    } else {
                        // We limit the index-th dimension (minus alphasSize to scale in a 0-S range)
                        boundary[index - alphasSize] = 0.0;
                    }
                }
                m.row(counter) = boundary;
                b[counter] = 1.0;
                ++counter;

                // Note that we only need to consider the first "counter" rows,
                // as the boundaries get merged in a single one.
                result = m.topRows(counter).colPivHouseholderQr().solve(b.head(counter));

                b[counter-1] = 0.0;

                // Add to found only if valid, otherwise skip.
                const double max = result.head(S).maxCoeff();
                if ((result.head(S).array() >= 0).all() && (max < 1.0) && checkDifferentSmall(max, 1.0)) {
                    vertices.first.emplace_back(result.head(S));
                    vertices.second.emplace_back(result[S]);
                }

                // Advance, and take the id of the first index changed in the
                // next combination.
                last = enumerator.advance();

                // If the index went over the alpha list, then we'd only have
                // boundaries, but we don't care about those cases (since we
                // assume we already have the corners of the simplex computed).
                // Thus, terminate.
                if ((*enumerator)[0] >= alphasSize) break;
            }
        }
        return vertices;
    }

    /**
     * @brief This function returns all vertices for a given range of planes.
     *
     * This function is used as a convenience to avoid duplicate plane problems.
     * It will still possibly return duplicate vertices though.
     *
     * @param range The range of planes to examine.
     * @param p A projection function to call on the iterators of the range (defaults to identity).
     *
     * @return A non-unique list of all the vertices found.
     */
    template <typename Range, typename P = identity>
    PointSurface findVerticesNaive(const Range & range, P p = P{}) {
        PointSurface retval;

        std::array<size_t, 1> indexToSkip;

        auto goodBegin = range.cbegin();
        for (size_t i = 0; i < range.size(); ++i, ++goodBegin) {
            // For each alpha, we find its vertices against the others.
            indexToSkip[0] = i;
            IndexSkipMap map(&indexToSkip, range);

            const auto cbegin = map.cbegin();
            const auto cend   = map.cend();

            // Note that the range we pass here is made by a single vector.
            auto newVertices = findVerticesNaive(goodBegin, goodBegin + 1, cbegin, cend, p, p);

            retval.first.insert(std::end(retval.first),
                std::make_move_iterator(std::begin(newVertices.first)),
                std::make_move_iterator(std::end(newVertices.first))
            );
            retval.second.insert(std::end(retval.second),
                std::make_move_iterator(std::begin(newVertices.second)),
                std::make_move_iterator(std::end(newVertices.second))
            );
        }
        return retval;
    }

    /**
     * @brief This function computes the optimistic value of a point given known vertices and values.
     *
     * This function computes an LP to determine the best possible value of a
     * point given all known best vertices around it.
     *
     * This function is needed in multi-objective settings (rather than
     * POMDPs), since the step where we compute the optimal value for a given
     * point is extremely expensive (it requires solving a full MDP). Thus
     * linear programming is used in order to determine an optimistic bound
     * when deciding the next point to extract from the queue during the linear
     * support process.
     *
     * Note that the input is the same as a PointSurface; the two components
     * have been kept as separate arguments simply to allow more freedom to the
     * user.
     *
     * @param p The point where we want to compute the best possible value.
     * @param points The points that make up the surface.
     * @param values The respective values of the input points.
     *
     * @return The best possible value that the input point can have given the known vertices.
     */
    double computeOptimisticValue(const Point & p, const std::vector<Point> & points, const std::vector<double> & values);

    /**
     * @brief This function computes the exact value of the input Point w.r.t. the given surfaces.
     *
     * The input CompactHyperplanes are used as an easy upper bound.
     *
     * Then, a linear programming is created that uses the input PointSurface.
     * What happens is that the linear program uses each Point (and its value)
     * to construct a piecewise linear surface, where the value of the input
     * belief is determined.
     *
     * The higher of the two surfaces is then returned as the value of the
     * input Point.
     *
     * @param p The point to compute the value of.
     * @param ubQ A set of Hyperplanes to use as a baseline surface.
     * @param ubV A set of Points (not on the corners of the simplex) to use as main interpolation.
     *
     * @return The value of the Point, and a vector containing the proportion in which each Point in the PointSurface contributes to the upper bound.
     */
    std::tuple<double, Vector> LPInterpolation(const Point & p, const CompactHyperplanes & ubQ, const PointSurface & ubV);

    /**
     * @brief This function computes an approximate, but quick, upper bound on the surface value at the input point.
     *
     * The input CompactHyperplanes are used as an easy upper bound.
     *
     * We then start to consider every surface composed by one Point in the
     * input PointSurface, and N-1 corners of the simplex (the highest corners
     * of the surface, as identified by the CompactHyperplanes). Since each
     * Point defines N such surfaces (one for each corner we "skip"), the
     * enumeration can be done fairly quickly.
     *
     * The overall surface has a sawtooth shape, from which the name of this
     * method. The approximation is not perfect, but some methods must use it
     * as using the LPInterpolation method would be too computationally
     * expensive.
     *
     * @param p The point to compute the value of.
     * @param ubQ A set of Hyperplanes to use as a baseline surface.
     * @param ubV A set of Points (not on the corners of the simplex) to use as main interpolation.
     *
     * @return The value of the Point, and a vector containing the proportion in which each Point in the PointSurface contributes to the upper bound.
     */
    std::tuple<double, Vector> sawtoothInterpolation(const Point & p, const CompactHyperplanes & ubQ, const PointSurface & ubV);

    /**
     * @brief This class implements an easy interface to do Witness discovery through linear programming.
     *
     * Witness discovery is the process of determining whether a given
     * Hyperplane is higher than any other; and if so, where.
     *
     * This class is meant to help finding witness points by solving the linear
     * programming needed. As such, it contains a linear programming problem where
     * constraints can be set. This class automatically sets the simplex constraint,
     * where a found Point's coordinates need to sum up to one.
     *
     * Optimal constraints can be progressively added as soon as found. When a
     * new constraint needs to be tested to see if a witness is available, the
     * findWitness() function can be called.
     */
    class WitnessLP {
        public:
            /**
             * @brief Basic constructor.
             *
             * This initializes lp_solve structures.
             *
             * @param S The number of corners of the simplex.
             */
            WitnessLP(size_t S);

            /**
             * @brief This function adds a new optimal constraint to the LP, which will not be removed unless the LP is reset.
             *
             * This function is used to add the optimal hyperplanes.
             *
             * @param v The optimal Hyperplane to add.
             */
            void addOptimalRow(const Hyperplane & v);

            /**
             * @brief This function solves the currently set LP.
             *
             * This function tries to solve the underlying LP, and if
             * successful returns the witness point which satisfies
             * the solution.
             *
             * @param v The Hyperplane to test against the optimal ones already added.
             *
             * @return If found, the Point witness to the set problem.
             */
            std::optional<Point> findWitness(const Hyperplane & v);

            /**
             * @brief This function resets the internal LP to only the simplex constraint.
             *
             * This function does not mess with the already allocated memory.
             */
            void reset();

            /**
             * @brief This function reserves space for a certain amount of rows (not counting the simplex) to avoid reallocations.
             *
             * @param rows The max number of constraints for the LP.
             */
            void allocate(size_t rows);

        private:
            size_t S;
            LP lp_;
    };
}

#endif
