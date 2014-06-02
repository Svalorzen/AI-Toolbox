#include <AIToolbox/POMDP/Policies/Policy.hpp>

#include <AIToolbox/POMDP/Utils.hpp>

namespace AIToolbox {
    namespace POMDP {
        Policy::Policy(size_t s, size_t a, size_t o, const ValueFunction & v) : PolicyInterface<Belief>(s, a), O(o), H(v.size()), policy_(v) {
            if ( H < 1 ) throw std::invalid_argument("The ValueFunction supplied to POMDP::Policy is empty.");
        }

        size_t Policy::sampleAction(const Belief & b) const {
            // We use the latest horizon here.
            auto & vlist = policy_.back();

            auto bestMatch = findBestAtBelief(getS(), b, std::begin(vlist), std::end(vlist));

            return std::get<ACTION>(*bestMatch);
        }

        std::tuple<size_t, size_t> Policy::sampleAction(const Belief & b, unsigned horizon) const {
            auto & vlist = policy_[horizon]; 

            auto begin     = std::begin(vlist);
            auto bestMatch = findBestAtBelief(getS(), b, begin, std::end(vlist));

            size_t action = std::get<ACTION>(*bestMatch);
            size_t id     = std::distance(begin, bestMatch);

            return std::make_tuple(action, id);
        }

        std::tuple<size_t, size_t> Policy::sampleAction(size_t id, size_t o, unsigned horizon) const {
            // Horizon + 1 means one step in the past.
            auto & vlist = policy_[horizon+1];

            size_t newId  = std::get<OBS>(vlist[id])[o];
            size_t action = std::get<ACTION>(policy_[horizon][newId]);

            return std::make_tuple(action, newId);
        }

        double Policy::getActionProbability(const Belief & b, size_t a) const {
            // At the moment we know that only one action is possible..
            size_t trueA = sampleAction(b);

            return ( a == trueA ? 1.0 : 0.0 );
        }

        double Policy::getActionProbability(const Belief & b, size_t a, unsigned horizon) const {
            // At the moment we know that only one action is possible..
            size_t trueA = std::get<0>(sampleAction(b, horizon));

            return ( a == trueA ? 1.0 : 0.0 );
        }

        size_t Policy::getO() const {
            return O;
        }

        size_t Policy::getH() const {
            return H;
        }
    }
}
