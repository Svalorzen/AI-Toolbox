#include <AIToolbox/POMDP/Utils.hpp>

namespace AIToolbox::POMDP {
    ValueFunction makeValueFunction(const size_t S) {
        auto values = MDP::Values(S);
        values.fill(0.0);

        return ValueFunction(1, VList(1, std::make_tuple(values, 0, VObs())));
    }

    bool operator<(const VEntry & lhs, const VEntry & rhs) {
        auto cmp = veccmp(std::get<0>(lhs), std::get<0>(rhs));
        if (cmp != 0) return cmp < 0;
        if (std::get<1>(lhs) != std::get<1>(rhs))
            return std::get<1>(lhs) < std::get<1>(rhs);
        return std::get<2>(lhs) < std::get<2>(rhs);
    }

    double weakBoundDistance(const VList & oldV, const VList & newV) {
        // Here we implement a weak bound (can also be seen in Cassandra's code)
        // This is mostly because a strong bound is more costly (it requires performing
        // multiple LPs) and also the code at the moment does not support it cleanly, so
        // I prefer waiting until I have a good implementation of an LP class that hides
        // complexity from here.
        //
        // The logic of the weak bound is the following: the variation between the old
        // VList and the new one is equal to the maximum distance between a ValueFunction
        // in the old VList with its closest match in the new VList. So the farthest from
        // closest.
        //
        // We define distance between two ValueFunctions as the maximum between their
        // element-wise difference.
        if ( !oldV.size() ) return 0.0;

        double distance = 0.0;
        for ( const auto & newVE : newV ) {
            // Initialize closest distance for newVE as infinity
            double closestDistance = std::numeric_limits<double>::infinity();
            for ( const auto & oldVE : oldV ) {
                // Compute the distance, we pick the max
                double distance = (std::get<VALUES>(newVE) - std::get<VALUES>(oldVE)).cwiseAbs().maxCoeff();

                // Keep the closest, we pick the min
                closestDistance = std::min(closestDistance, distance);
            }
            // Keep the maximum distance between a new VList and its closest old VList
            distance = std::max(distance, closestDistance);
        }
        return distance;
    }
}
