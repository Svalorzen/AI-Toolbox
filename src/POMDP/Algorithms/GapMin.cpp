#include <AIToolbox/POMDP/Algorithms/GapMin.hpp>

#include <AIToolbox/Utils/LP.hpp>

namespace AIToolbox::POMDP {
    GapMin::GapMin(const double initialTolerance, const unsigned digits) :
        precisionDigits_(digits)
    {
        setInitialTolerance(initialTolerance);
    }

    void GapMin::setInitialTolerance(double initialTolerance) {
        if ( initialTolerance < 0.0 ) throw std::invalid_argument("Initial tolerance must be >= 0");
        initialTolerance_ = initialTolerance;
    }

    double GapMin::getInitialTolerance() const {
        return initialTolerance_;
    }

    void GapMin::setPrecisionDigits(unsigned digits) {
        precisionDigits_ = digits;
    }

    unsigned GapMin::getPrecisionDigits() const {
        return precisionDigits_;
    }

    bool GapMin::QueueElementLess::operator() (const QueueElement& arg1, const QueueElement& arg2) const
    {
        return std::get<1>(arg1) < std::get<1>(arg2);
    }

    void GapMin::cleanUp(const MDP::QFunction & ubQ, UbVType * ubVp, Matrix2D * fibQp) {
        assert(ubVp);
        assert(fibQp);

        UbVType & ubV = *ubVp;
        Matrix2D & fibQ = *fibQp;

        if (ubV.first.size() == 1) return;

        std::vector<size_t> toRemove;
        size_t i = ubV.first.size();

        // For each belief, we try to compute its upper bound with the others.
        // If it doesn't change, it means that we don't really need it, and
        // thus we remove it.
        do {
            --i;

            std::swap(ubV.first[i], ubV.first.back());
            std::swap(ubV.second[i], ubV.second.back());

            auto belief = std::move(ubV.first.back());
            auto value = ubV.second.back();

            ubV.first.pop_back();
            ubV.second.pop_back();

            const auto [v, dist] = UB(belief, ubQ, ubV);
            (void)dist;

            if (value >= v - tolerance_) {
                toRemove.push_back(i);
            } else {
                // Unpop and unswap, since we need to keep the order consistent
                // (fibQ depends on it). This could be done with a couple less
                // moves but like this it's more clear.
                ubV.first.emplace_back(std::move(belief));
                ubV.second.emplace_back(value);

                std::swap(ubV.first[i], ubV.first.back());
                std::swap(ubV.second[i], ubV.second.back());
            }
        } while (i != 0 && ubV.first.size() > 1);
        // If all beliefs are useful, we're done.
        if (toRemove.size() == 0) return;

        // Here we do a bit of fancy dance in order to do as little operations
        // as possible to remove the unneeded rows from fibQ. Additionally, we
        // try to also do as few block operations as possible, in the hope that
        // Eigen gets to optimize those and thus speeds us up.
        //
        // We need to allocate another matrix anyway since if we copied from
        // fibQ to fibQ we'd have to make an allocation per copy anyway. So we
        // do only one and we're done.
        //
        // The idea is simply copying one block at a time towards the beginning.
        const size_t S = ubQ.rows();
        const size_t oldRows = fibQ.rows();
        const size_t newRows = oldRows - toRemove.size();
        const size_t cols = fibQ.cols();

        Matrix2D newFibQ(newRows, cols);

        // Copy the "default" part of the matrix (which is the same as ubQ at the moment)
        newFibQ.topRows(S).noalias() = fibQ.topRows(S);

        size_t beginSource = 0; // This is not S because we use it to check against the IDs to remove.
        size_t beginTarget = S;
        size_t toCopy;

        i = toRemove.size();
        while (i > 0) {
            while (i > 0 && toRemove[i-1] == beginSource) --i, ++beginSource;

            if (i == 0) {
                if (beginSource > oldRows - S) break;
                toCopy = oldRows - S - beginSource;
            } else {
                toCopy = toRemove[i-1] - beginSource;
            }

            newFibQ.block(beginTarget, 0, toCopy, cols).noalias() = fibQ.block(beginSource + S, 0, toCopy, cols);

            beginTarget += toCopy;
            beginSource += toCopy;
        }
        // And we give it back
        fibQ = std::move(newFibQ);
    }

    std::tuple<double, Vector> GapMin::UB(const Belief & belief, const MDP::QFunction & ubQ, const UbVType & ubV) {
        // Here we find all beliefs that have the same "zeroes" as the input one.
        // This is done to reduce the amount of work the LP has to do.
        std::vector<size_t> zeroStates;
        std::vector<size_t> nonZeroStates;
        for (size_t s = 0; s < static_cast<size_t>(belief.size()); ++s) {
            if (checkEqualSmall(belief[s], 0.0))
                zeroStates.push_back(s);
            else
                nonZeroStates.push_back(s);
        }

        std::vector<size_t> compatibleBeliefs;
        if (zeroStates.size() == 0) {
            // If no zero states, we match all beliefs.
            compatibleBeliefs.resize(ubV.first.size());
            std::iota(std::begin(compatibleBeliefs), std::end(compatibleBeliefs), 0);
        } else {
            for (size_t i = 0; i < ubV.first.size(); ++i) {
                bool add = true;
                for (const auto s : zeroStates) {
                    if (checkDifferentSmall(ubV.first[i][s], 0.0)) {
                        add = false;
                        break;
                    }
                }
                if (add) compatibleBeliefs.push_back(i);
            }
        }

        // If there's no other belief on the same plane as this one, the V can't
        // help us with the bound. So we just use the Q, and we copy its values in the
        // corners of the belief.
        if (compatibleBeliefs.size() == 0) {
            Vector retval(belief.size() + ubV.first.size());

            retval.head(belief.size()).noalias() = belief;
            retval.tail(ubV.first.size()).setZero();

            return std::make_tuple((belief.transpose() * ubQ).maxCoeff(), std::move(retval));
        }

        Vector cornerVals = ubQ.rowwise().maxCoeff();

        double unscaledValue;
        Vector result;

        // If there's only a single compatible belief, we don't really need to run
        // an LP.
        if (compatibleBeliefs.size() == 1) {
            const auto & compBelief = ubV.first[compatibleBeliefs[0]];

            result.resize(1);
            result[0] = (belief.cwiseQuotient(compBelief)).minCoeff();

            unscaledValue = result[0] * (ubV.second[compatibleBeliefs[0]] - compBelief.transpose() * cornerVals);
        } else {
            /*
             * Here we run the LP.
             *
             * In order to obtain the linear approximation for the upper bound of
             * the input belief, given that we already know the values for a set of
             * beliefs, we need to solve an LP in the form:
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
             * belief (bin), found by interpolating all other known beliefs. At the
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
             * - Nonzero & Compatible Beliefs
             *
             * If the input belief is restricted to a subset of dimensions in the
             * VFunction (meaning some of its values are zero), and we have beliefs
             * in that exact same subset, we can just use those in order to determine
             * the upper bound of the input. This is true since additional dimensions
             * won't affect the ValueFunction in the particular subspace the input
             * belief is in. All other beliefs are discarded. Note that we'll need
             * to fill in zeroes for the coefficients of the discarded beliefs after
             * we are done.
             *
             * - Removal of Corner Values
             *
             * Ideally, one would want the corner beliefs/values to be included in
             * the list of beliefs to use for interpolation, since they are needed.
             * However, all other belief values can simply be scaled down as if the
             * corner values were zero, and the resulting solution would not change.
             * The only thing is that the values obtained for the target function
             * would need to be scaled back before being returned by the LP.
             *
             * The other thing is that removing the corner values from the
             * beliefs also means removing the coefficients for them in the LP.
             * This is good as the LP is simpler, but pretty much makes it
             * infeasible. So what we do is change the constraints, and instead
             * of making them equal, we make them less or equal.
             *
             * In the end we're going to fix all missing numbers anyway by
             * filling the corners with the needed numbers.
             */

            // We're going to have one column per compatible belief (plus one, but
            // that's implied).
            LP lp(compatibleBeliefs.size() + 1);
            lp.resize(nonZeroStates.size() + 1); // One row per state, plus the K constraint.
            size_t i;

            // Goal: minimize K.
            lp.setObjective(compatibleBeliefs.size(), false);

            // IMPORTANT: K is unbounded, since the value function may be negative.
            lp.setUnbounded(compatibleBeliefs.size());

            // By default we don't have K, only at the end.
            lp.row[compatibleBeliefs.size()] = +0.0;

            // So each row contains the same-index element from all the compatible
            // beliefs, and they should sum up to that same element in the input belief.
            for (const auto s : nonZeroStates) {
                i = 0;
                for (const auto b : compatibleBeliefs)
                    lp.row[i++] = ubV.first[b][s];
                lp.pushRow(LP::Constraint::LessEqual, belief[s]);
            }

            // Finally we setup the last row.
            i = 0;
            for (const auto b : compatibleBeliefs) {
                double val = ubV.second[b];
                for (const auto s : nonZeroStates)
                    val -= ubV.first[b][s] * cornerVals[s];
                lp.row[i++] = val;
            }
            lp.row[i] = -1.0;
            lp.pushRow(LP::Constraint::Equal, 0.0);

            // Now solve
            auto tmp = lp.solve(compatibleBeliefs.size(), &unscaledValue);
            if (!tmp)
                throw std::runtime_error("GapMin UB process failed!");
            result = *tmp;
        }

        // We scale back the value as if we had considered the corners.
        double ubValue = unscaledValue + belief.transpose() * cornerVals;

        Vector retval(belief.size() + ubV.first.size());
        retval.setZero();

        // And we fix the coefficients in order to actually make the equalities
        // hold.
        for (const auto s : nonZeroStates) {
            double sum = 0.0;
            for (size_t i = 0; i < compatibleBeliefs.size(); ++i)
                sum += ubV.first[compatibleBeliefs[i]][s] * result[i];
            retval[s] = belief[s] - sum;
        }
        retval.tail(compatibleBeliefs.size()) = result;
        // Remove infinitesimal/negative values
        for (auto i = 0; i < retval.size(); ++i)
            if (checkEqualSmall(retval[i], 0.0) || retval[i] < 0.0) retval[i] = 0.0;

        return std::make_tuple(ubValue, std::move(retval));
    }
}
