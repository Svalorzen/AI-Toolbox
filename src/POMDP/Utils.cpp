#include <AIToolbox/POMDP/Utils.hpp>

namespace AIToolbox {
    namespace POMDP {

        VEntry makeVEntry(size_t S, size_t a, size_t O) {
            return std::make_tuple(MDP::Values(S, 0.0), a, VObs(O, 0));
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
            MDP::Values helper(std::get<VALUES>(oldV[0]).size()); // We use this to compute differences.
            auto hBegin = std::begin(helper), hEnd = std::end(helper);

            double distance = 0.0;
            for ( auto & newVE : newV ) {
                auto nBegin = std::begin(std::get<0>(newVE)), nEnd = std::end(std::get<0>(newVE));

                double closestDistance = std::numeric_limits<double>::infinity();
                for ( auto & oldVE : oldV ) {
                    auto computeVariation = [](double lhs, double rhs) { return std::fabs(lhs - rhs); };
                    std::transform(nBegin, nEnd, std::begin(std::get<0>(oldVE)), hBegin, computeVariation );

                    // Compute the distance, we pick the max
                    double distance = *std::max_element(hBegin, hEnd);

                    // Keep the closest, we pick the min
                    closestDistance = std::min(closestDistance, distance);
                }
                // Keep the maximum distance between a new VList and its closest old VList
                distance = std::max(distance, closestDistance);
            }
            return distance;
        }
    }
}
