#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>

namespace AIToolbox::POMDP {
    IncrementalPruning::IncrementalPruning(const unsigned h, const double t) :
            horizon_(h)
    {
        setTolerance(t);
    }

    void IncrementalPruning::setHorizon(const unsigned h) {
        horizon_ = h;
    }
    void IncrementalPruning::setTolerance(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Tolerance must be >= 0");
        tolerance_ = t;
    }

    unsigned IncrementalPruning::getHorizon() const {
        return horizon_;
    }

    double IncrementalPruning::getTolerance() const {
        return tolerance_;
    }

    VList IncrementalPruning::crossSum(const VList & l1, const VList & l2, const size_t a, const bool order) {
        VList c;

        if ( !(l1.size() && l2.size()) ) return c;

        // We can get the sizes of the observation vectors
        // outside since all VEntries for our input VLists
        // are guaranteed to be sized equally.
        const auto O1size  = l1[0].observations.size();
        const auto O2size  = l2[0].observations.size();
        for ( const auto & v1 : l1 ) {
            const auto O1begin = std::begin(v1.observations);
            const auto O1end   = std::end  (v1.observations);
            for ( const auto & v2 : l2 ) {
                const auto O2begin = std::begin(v2.observations);
                const auto O2end   = std::end  (v2.observations);
                // Cross sum
                auto v = v1.values + v2.values;

                // This step now depends on which order the two lists
                // are. This function is only used in this class, so we
                // know that the two lists are "adjacent"; however one
                // is after the other. `order` tells us which one comes
                // first, and we join the observation vectors accordingly.
                VObs obs; obs.reserve(O1size + O2size);
                if ( order ) {
                    obs.insert(std::end(obs), O1begin, O1end);
                    obs.insert(std::end(obs), O2begin, O2end);
                } else {
                    obs.insert(std::end(obs), O2begin, O2end);
                    obs.insert(std::end(obs), O1begin, O1end);
                }
                c.emplace_back(std::move(v), a, std::move(obs));
            }
        }

        return c;
    }
}
