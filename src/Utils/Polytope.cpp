#include <AIToolbox/Utils/Polytope.hpp>

namespace AIToolbox {
    double computeOptimisticValue(const Point & p, const std::vector<Point> & points, const std::vector<double> & values) {
        assert(points.size() == values.size());

        if (points.size() == 0) return 0.0;
        const size_t S = p.size();

        LP lp(S);

        /*
         * With this LP we are looking for an optimistic hyperplane that can
         * tightly fit all corners that we already have, and maximize the value
         * at the input point.
         *
         * Our constraints are of the form
         *
         * vertex[0][0]) * h0 + vertex[0][1]) * h1 + ... <= vertex[0].currentValue
         * vertex[1][0]) * h0 + vertex[1][1]) * h1 + ... <= vertex[1].currentValue
         * ...
         *
         * Since we are looking for an optimistic hyperplane, all variables are
         * unbounded since the hyperplane may need to go negative at some
         * states.
         *
         * Finally, our constraint is a row to maximize:
         *
         * b * v0 + b * v1 + ...
         *
         * Which means we try to maximize the value of the input point with the
         * newly found hyperplane.
         */

        // Set objective to maximize
        lp.row = p;
        lp.setObjective(true);

        // Set unconstrained to all variables
        for (size_t s = 0; s < S; ++s)
            lp.setUnbounded(s);

        // Set constraints for all input points and current values.
        for (size_t i = 0; i < points.size(); ++i) {
            lp.row = points[i];
            lp.pushRow(LP::Constraint::LessEqual, values[i]);
        }

        double retval;
        // Note that we don't care about the optimistic alphavector, so we
        // discard it. We check that everything went fine though, in theory
        // there shouldn't be any problems here.
        auto solution = lp.solve(0, &retval);
        assert(solution);

        return retval;
    }

    std::tuple<double, Vector> LPInterpolation(const Point & point, const CompactHyperplanes & ubQ, const PointSurface & ubV) {
        // Here we find all points that have the same "zeroes" as the input one.
        // This is done to reduce the amount of work the LP has to do.
        std::vector<size_t> zeroStates;
        std::vector<size_t> nonZeroStates;
        for (size_t s = 0; s < static_cast<size_t>(point.size()); ++s) {
            if (checkEqualSmall(point[s], 0.0))
                zeroStates.push_back(s);
            else
                nonZeroStates.push_back(s);
        }

        std::vector<size_t> compatiblePoints;
        if (zeroStates.size() == 0) {
            // If no zero states, we match all points.
            compatiblePoints.resize(ubV.first.size());
            std::iota(std::begin(compatiblePoints), std::end(compatiblePoints), 0);
        } else {
            for (size_t i = 0; i < ubV.first.size(); ++i) {
                bool add = true;
                for (const auto s : zeroStates) {
                    if (checkDifferentSmall(ubV.first[i][s], 0.0)) {
                        add = false;
                        break;
                    }
                }
                if (add) compatiblePoints.push_back(i);
            }
        }

        // If there's no other point on the same plane as this one, the V can't
        // help us with the bound. So we just use the Q, and we copy its values in the
        // corners of the point.
        if (compatiblePoints.size() == 0) {
            Vector retval(point.size() + ubV.first.size());

            retval.head(point.size()).noalias() = point;
            retval.tail(ubV.first.size()).setZero();

            return std::make_tuple((point.transpose() * ubQ).maxCoeff(), std::move(retval));
        }

        const Vector cornerVals = ubQ.rowwise().maxCoeff();

        double unscaledValue;
        Vector result;

        // If there's only a single compatible point, we don't really need to run
        // an LP.
        if (compatiblePoints.size() == 1) {
            const auto & compPoint = ubV.first[compatiblePoints[0]];

            result.resize(1);
            result[0] = (point.cwiseQuotient(compPoint)).minCoeff();

            unscaledValue = result[0] * (ubV.second[compatiblePoints[0]] - compPoint.transpose() * cornerVals);
        } else {
            /*
             * Here we run the LP.
             *
             * In order to obtain the linear approximation for the upper bound of
             * the input point, given that we already know the values for a set of
             * points, we need to solve an LP in the form:
             *
             * c[0] * b[0][0] + c[1] * b[1][0] + ...                = bin[0]
             * c[0] * b[0][1] + c[1] * b[1][1] + ...                = bin[1]
             * c[0] * b[0][2] + c[1] * b[1][2] + ...                = bin[2]
             * ...
             * c[0] * v[0]    + c[1] * v[1]    + ... - K            = 0
             *
             * And we minimize K to get:
             *
             * argmin(c) = sum( c * v ) = K
             *
             * This way K will be the minimum upper bound possible for the input
             * point (bin), found by interpolating all other known points. At the
             * same time we apply the linear approximation by enforcing
             *
             * sum( c * b ) = bin
             *
             * We also set each c to be >= 0.
             *
             * OPTIMIZATIONS:
             *
             * Once we have defined the problem, we can apply a series of
             * optimizations to reduce the size of the LP to be solved. These were
             * taken from the MATLAB code published for the GapMin algorithm. Written
             * interpretation below is mine.
             *
             * - Nonzero & Compatible Points
             *
             * If the input point is restricted to a subset of dimensions in the
             * VFunction (meaning some of its values are zero), and we have points
             * in that exact same subset, we can just use those in order to determine
             * the upper bound of the input. This is true since additional dimensions
             * won't affect the ValueFunction in the particular subspace the input
             * point is in. All other points are discarded. Note that we'll need
             * to fill in zeroes for the coefficients of the discarded points after
             * we are done.
             *
             * - Removal of Corner Values
             *
             * Ideally, one would want the corner points/values to be included in
             * the list of points to use for interpolation, since they are needed.
             * However, all other point values can simply be scaled down as if the
             * corner values were zero, and the resulting solution would not change.
             * The only thing is that the values obtained for the target function
             * would need to be scaled back before being returned by the LP.
             *
             * The other thing is that removing the corner values from the
             * points also means removing the coefficients for them in the LP.
             * This is good as the LP is simpler, but pretty much makes it
             * infeasible. So what we do is change the constraints, and instead
             * of making them equal, we make them less or equal.
             *
             * In the end we're going to fix all missing numbers anyway by
             * filling the corners with the needed numbers.
             */

            // We're going to have one column per compatible point (plus one, but
            // that's implied).
            LP lp(compatiblePoints.size() + 1);
            lp.resize(nonZeroStates.size() + 1); // One row per state, plus the K constraint.
            size_t i;

            // Goal: minimize K.
            lp.setObjective(compatiblePoints.size(), false);

            // IMPORTANT: K is unbounded, since the value function may be negative.
            lp.setUnbounded(compatiblePoints.size());

            // By default we don't have K, only at the end.
            lp.row[compatiblePoints.size()] = +0.0;

            // So each row contains the same-index element from all the compatible
            // points, and they should sum up to that same element in the input point.
            for (const auto s : nonZeroStates) {
                i = 0;
                for (const auto b : compatiblePoints)
                    lp.row[i++] = ubV.first[b][s];
                lp.pushRow(LP::Constraint::LessEqual, point[s]);
            }

            // Finally we setup the last row.
            i = 0;
            for (const auto b : compatiblePoints) {
                double val = ubV.second[b];
                for (const auto s : nonZeroStates)
                    val -= ubV.first[b][s] * cornerVals[s];
                lp.row[i++] = val;
            }
            lp.row[i] = -1.0;
            lp.pushRow(LP::Constraint::Equal, 0.0);

            // Now solve
            auto tmp = lp.solve(compatiblePoints.size(), &unscaledValue);
            if (!tmp)
                throw std::runtime_error("GapMin UB process failed!");
            result = *tmp;
        }

        // We scale back the value as if we had considered the corners.
        double ubValue = unscaledValue + point.transpose() * cornerVals;

        Vector retval(point.size() + ubV.first.size());
        retval.setZero();

        // And we fix the coefficients in order to actually make the equalities
        // hold.
        for (const auto s : nonZeroStates) {
            double sum = 0.0;
            for (size_t i = 0; i < compatiblePoints.size(); ++i)
                sum += ubV.first[compatiblePoints[i]][s] * result[i];
            retval[s] = point[s] - sum;
        }
        retval.tail(compatiblePoints.size()) = result;
        // Remove infinitesimal/negative values
        for (auto i = 0; i < retval.size(); ++i)
            if (checkEqualSmall(retval[i], 0.0) || retval[i] < 0.0) retval[i] = 0.0;

        return std::make_tuple(ubValue, std::move(retval));
    }

    std::tuple<double, Vector> sawtoothInterpolation(const Point & point, const CompactHyperplanes & ubQ, const PointSurface & ubV) {
        // Compute top surface on simplex corners.
        const Vector cornerVals = ubQ.rowwise().maxCoeff();

        // Cache zero elements since checkEqualSmall is somewhat expensive.
        std::vector<char> zeroStates(point.size());
        for (size_t s = 0; s < zeroStates.size(); ++s)
            zeroStates[s] = checkEqualSmall(point[s], 0.0);

        // For each point, we are going to find out whether there is a
        // combination of corners which can give us an approximation of the
        // input point's value. Obviously we are going to pick the lowest, as
        // this function is computing an upper bound.
        size_t minI = 0;
        double minCF = 0.0, minC;
        for (size_t i = 0; i < ubV.first.size(); ++i) {
            // This finds the corner of the simplex we can "skip" in order to
            // obtain the lowest surface possible at the input point.
            // We need to skip the corners where this point is zero, as it
            // can't give us any real information there.
            double c = std::numeric_limits<double>::max();
            for (size_t s = 0; s < zeroStates.size(); ++s) {
                const bool isThisZero = checkEqualSmall(ubV.first[i][s], 0.0);
                // If the input point is zero and this point is not, then it
                // cannot help us at all.
                if (zeroStates[s] && !isThisZero)
                    goto next;
                // If this is zero, then the ratio would be infinite - hardly
                // helping.
                if (isThisZero)
                    continue;
                c = std::min(c, point[s] / ubV.first[i][s]);
            }
            // Sanity check just in case
            // If we are here we should have found something
            // And that something should not be higher than 1.0
            assert((c < 1.0) | checkEqualSmall(c, 1.0));
            c = std::min(c, 1.0);
            // We need this scope to avoid complaints of the goto.
            {
                // This represents the ratio times distance between this point
                // and the surface between the simplex corners (i.e. how much
                // we can reduce the input's point value from the naive simplex
                // surface).
                const auto cf = c * (ubV.second[i] - ubV.first[i].dot(cornerVals));
                if (cf < minCF) {
                    minC = c;
                    minCF = cf;
                    minI = i;
                }
            }
next:;
        }
        // Naive height
        const auto basicV = (point.transpose() * ubQ).maxCoeff();
        // Sawtooth height (note that minCF is negative)
        const auto v = point.dot(cornerVals) + minCF;

        Vector retval(point.size() + ubV.first.size());
        // Set to zero all coefficients for the beliefs only, since we are
        // going to write in the corner ones anyway.
        retval.tail(ubV.first.size()).setZero();

        // If we didn't need the interpolation to begin with (maybe we didn't
        // find any points that can help us)..
        if (basicV < v) {
            retval.head(point.size()).noalias() = point;

            return std::make_tuple(basicV, std::move(retval));
        }

        retval.head(point.size()).noalias() = point - ubV.first[minI] * minC;
        retval[minI] = minC;

        return std::make_tuple(v, std::move(retval));
    }

    // -----------------------------------------------------

    WitnessLP::WitnessLP(const size_t s) : S(s), lp_(s+2)
    {
        /*
         * Here we setup the part of the lp that never changes (at least with this number of states)
         * Our constraints are of the form
         *
         * b0 >= 0
         * b1 >= 0
         * ...
         * b0 + b1 + ... + bn = 1.0
         * (v[0] - best[0][0]) * b0 + (v[1] - best[0][1]) * b1 + ... - delta >= 0
         * (v[0] - best[1][0]) * b0 + (v[1] - best[1][1]) * b1 + ... - delta >= 0
         * ...
         *
         * In particular we know that:
         *
         * 1) By default no variable will assume negative values, so
         * the only constraint that we need to put is the simplex one,
         * as lp_solve automatically sets all variables to be positive.
         *
         * 2) The simplex constraint is the only constraint that never changes,
         * so we are going to set it up now.
         *
         * 3) The other constraints can be rewritten as follows:
         *
         *       v[0] * b0 +       v[1] * b1 + ... - K          = 0
         * best[0][0] * b0 + best[0][1] * b1 + ... - K + delta <= 0
         * best[1][0] * b0 + best[1][1] * b1 + ... - K + delta <= 0
         * ...
         *
         * Where basically with the first constraint we are setting K to the
         * value of v in the final point, and we are forcing all other values
         * to be less than that by forcing:
         *
         * delta > 0
         *
         * The client can then add these and remove them at their discretion.
         */

        // Goal: maximize delta.
        lp_.setObjective(S+1, true);

        // CONSTRAINT: This is the simplex constraint (coordinates sum to 1)
        {
            // Note: lp_solve reads elements from 1 onwards, so we don't set row[0]
            for ( size_t i = 0; i < S; ++i )
                lp_.row[i] = 1.0;
            lp_.row[S]     = 0.0; // magic coefficient
            lp_.row[S + 1] = 0.0; // delta coefficient
            // The cols value doesn't really do anything here, the whole row is read
            lp_.pushRow(LP::Constraint::Equal, 1.0);
        }

        // IMPORTANT: K is unbounded, since the value function may be negative.
        lp_.setUnbounded(S);

        lp_.row[S]     = -1.0;
        lp_.row[S + 1] = +0.0;
    }

    void WitnessLP::addOptimalRow(const Hyperplane & v) {
        for ( size_t i = 0; i < S; ++i )
            lp_.row[i] = v[i];
        // Temporarily set the delta constraint
        lp_.row[S+1] = +1.0;
        lp_.pushRow(LP::Constraint::LessEqual, 0.0);

        lp_.row[S+1] = 0.0;
    }

    std::optional<Point> WitnessLP::findWitness(const Hyperplane & v) {
        // Add witness constraint
        for ( size_t i = 0; i < S; ++i )
            lp_.row[i] = v[i];
        lp_.pushRow(LP::Constraint::Equal, 0.0);

        double deltaValue;
        auto solution = lp_.solve(S, &deltaValue);

        // Remove it
        lp_.popRow();

        // We have found a witness point if we have found a point where the
        // value of the supplied hyperplane is greater than ALL others. Thus we
        // just need to verify that the variable we have maximixed is actually
        // greater than 0.
        if (deltaValue <= 0)
            solution.reset();

        return solution;
    }

    void WitnessLP::reset() {
        lp_.resize(1);
    }

    void WitnessLP::allocate(const size_t rows) {
        lp_.resize(rows+1);
    }
}
