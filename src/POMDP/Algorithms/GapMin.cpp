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

    void GapMin::cleanUp(const MDP::QFunction & ubQ, UpperBoundValueFunction * ubVp, Matrix2D * fibQp) {
        assert(ubVp);
        assert(fibQp);

        UpperBoundValueFunction & ubV = *ubVp;
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

            const auto [v, dist] = LPInterpolation(belief, ubQ, ubV);
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
}
