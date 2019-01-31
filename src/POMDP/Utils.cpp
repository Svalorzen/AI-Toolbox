#include <AIToolbox/POMDP/Utils.hpp>

namespace AIToolbox::POMDP {
    ValueFunction makeValueFunction(const size_t S) {
        auto values = MDP::Values(S);
        values.setZero();

        return ValueFunction(1, VList(1, {values, 0, VObs()}));
    }

    bool operator<(const VEntry & lhs, const VEntry & rhs) {
        auto cmp = veccmp(lhs.values, rhs.values);
        if (cmp != 0) return cmp < 0;
        if (lhs.action != rhs.action)
            return lhs.action < rhs.action;
        return lhs.observations < rhs.observations;
    }
    bool operator==(const VEntry & lhs, const VEntry & rhs) {
        return veccmp(lhs.values, rhs.values) == 0 &&
               lhs.action == rhs.action &&
               lhs.observations == rhs.observations;
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
                double distance = (newVE.values - oldVE.values).cwiseAbs().maxCoeff();

                // Keep the closest, we pick the min
                closestDistance = std::min(closestDistance, distance);
            }
            // Keep the maximum distance between a new VList and its closest old VList
            distance = std::max(distance, closestDistance);
        }
        return distance;
    }
}
