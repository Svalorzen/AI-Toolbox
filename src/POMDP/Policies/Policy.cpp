#include <AIToolbox/POMDP/Policies/Policy.hpp>

#include <AIToolbox/POMDP/Utils.hpp>

namespace AIToolbox::POMDP {
    Policy::Policy(const size_t s, const size_t a, const size_t o) :
            Base(s, a), O(o), H(0), policy_(makeValueFunction(S)) {}

    Policy::Policy(const size_t s, const size_t a, const size_t o, const ValueFunction & v) :
            Base(s, a), O(o), H(v.size()-1), policy_(v)
    {
        if ( !v.size() ) throw std::invalid_argument("The ValueFunction supplied to POMDP::Policy is empty.");
    }

    size_t Policy::sampleAction(const Belief & b) const {
        // We use the latest horizon here.
        const auto & vlist = policy_.back();

        const auto bestMatch = findBestAtPoint(b, std::begin(vlist), std::end(vlist), nullptr, unwrap);

        return bestMatch->action;
    }

    std::tuple<size_t, size_t> Policy::sampleAction(const Belief & b, const unsigned horizon) const {
        const auto & vlist = policy_[horizon];

        const auto bestMatch = findBestAtPoint(b, std::begin(vlist), std::end(vlist), nullptr, unwrap);

        const size_t action = bestMatch->action;
        const size_t id     = std::distance(std::begin(vlist), bestMatch);

        return std::make_tuple(action, id);
    }

    std::tuple<size_t, size_t> Policy::sampleAction(const size_t id, const size_t o, const unsigned horizon) const {
        // Horizon + 1 means one step in the past.
        // Note that the zero entry is never supposed to be used, and it's just
        // a byproduct of the computing process.
        const auto & vlist = policy_[horizon+1];

        const size_t newId  = vlist[id].observations[o];
        const size_t action = policy_[horizon][newId].action;

        return std::make_tuple(action, newId);
    }

    double Policy::getActionProbability(const Belief & b, const size_t & a) const {
        // At the moment we know that only one action is possible..
        const size_t trueA = sampleAction(b);

        return ( a == trueA ? 1.0 : 0.0 );
    }

    double Policy::getActionProbability(const Belief & b, const size_t a, const unsigned horizon) const {
        // At the moment we know that only one action is possible..
        const size_t trueA = std::get<0>(sampleAction(b, horizon));

        return ( a == trueA ? 1.0 : 0.0 );
    }

    size_t Policy::getO() const {
        return O;
    }

    size_t Policy::getH() const {
        return H;
    }

    const ValueFunction & Policy::getValueFunction() const {
        return policy_;
    }
}
