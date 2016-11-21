#include <AIToolbox/POMDP/Policies/Policy.hpp>

#include <AIToolbox/POMDP/Utils.hpp>

#include <iostream>

namespace AIToolbox {
    namespace POMDP {
        Policy::Policy(const size_t s, const size_t a, const size_t o) :
                PolicyInterface<Belief>(s, a), O(o), H(0), policy_(1, VList(1, makeVEntry(S))) {}

        Policy::Policy(const size_t s, const size_t a, const size_t o, const ValueFunction & v) :
                PolicyInterface<Belief>(s, a), O(o), H(v.size()-1), policy_(v)
        {
            if ( !v.size() ) throw std::invalid_argument("The ValueFunction supplied to POMDP::Policy is empty.");
        }

        size_t Policy::sampleAction(const Belief & b) const {
            // We use the latest horizon here.
            auto & vlist = policy_.back();

            auto bestMatch = findBestAtBelief(b, std::begin(vlist), std::end(vlist));

            return std::get<ACTION>(*bestMatch);
        }

        std::tuple<size_t, size_t> Policy::sampleAction(const Belief & b, const unsigned horizon) const {
            const auto & vlist = policy_[horizon];

            const auto begin     = std::begin(vlist);
            const auto bestMatch = findBestAtBelief(b, begin, std::end(vlist));

            const size_t action = std::get<ACTION>(*bestMatch);
            const size_t id     = std::distance(begin, bestMatch);

            return std::make_tuple(action, id);
        }

        std::tuple<size_t, size_t> Policy::sampleAction(const size_t id, const size_t o, const unsigned horizon) const {
            // Horizon + 1 means one step in the past.
            const auto & vlist = policy_[horizon+1];

            const size_t newId  = std::get<OBS>(vlist[id])[o];
            const size_t action = std::get<ACTION>(policy_[horizon][newId]);

            return std::make_tuple(action, newId);
        }

        double Policy::getActionProbability(const Belief & b, const size_t a) const {
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

        // IO

        std::ostream& operator<<(std::ostream &os, const Policy & p) {
            const auto & vf = p.getValueFunction();

            // VLists
            for ( size_t h = 1; h < vf.size(); ++h ) {
                const auto & vl = vf[h];
                // VEntries
                for ( auto & vv : vl ) {
                    // Values
                    os << std::get<VALUES>(vv).transpose() << ' ';
                    // Action
                    os << std::get<ACTION>(vv) << ' ';
                    // Obs
                    for ( auto & o : std::get<OBS>(vv) )
                        os << o << ' ';
                    os << '\n';
                }
                // Horizon separator
                os << "@\n";
            }
            // We close with a second at sign so that other things can also be
            // put on the stream, and the loader will work.
            os << "@\n";

            return os;
        }

        bool checkRemoveAtSign(std::istream &is) {
            char c = (is >> std::ws).peek();
            if ( c == '@' ) {
                is >> c;
                return true;
            }
            return false;
        }

        std::istream& operator>>(std::istream &is, Policy & p) {
            const size_t S = p.getS();
            const size_t A = p.getA();
            const size_t O = p.getO();

            // We automatically generate the horizon 0 entry.
            ValueFunction vf(1, VList(1, makeVEntry(S)));

            // This variable keeps track of allowed obs indeces.
            size_t oldH = 1;
            // This variable indicates whether we found an horizon separator.
            bool newHorizon = true;

            while ( true ) {
                if ( newHorizon ) {
                    // If we find a '@' here, we have finished.
                    if ( checkRemoveAtSign(is) )
                        break;

                    oldH = vf.back().size();
                    vf.emplace_back();
                    newHorizon = false;
                }

                MDP::Values values(S);
                size_t action;
                POMDP::VObs obs(O, 0);

                // Values
                for ( size_t i = 0; i < S; ++i )
                    if ( !(is >> values(i)) )
                        goto failure;
                // Action
                if ( !(is >> action) || action >= A ) {
                    std::cerr << "Read invalid value for actions from stream.\n";
                    goto failure;
                }
                // Obs
                for ( auto & o : obs ) {
                    if ( !(is >> o) || ( o >= oldH && oldH ) ) {
                        std::cerr << "Observation id exceeds size of previous horizon.\n";
                        goto failure;
                    }
                }

                vf.back().emplace_back(values, action, obs);

                // Check if next char after whitespace is a @ that
                // marks a new horizon.
                if ( checkRemoveAtSign(is) )
                    newHorizon = true;
            }

            p.H = vf.size() - 1;
            p.policy_ = std::move(vf);
            return is;

failure:
            std::cerr << "Could not read correctly from input stream.\n";
            is.setstate(std::ios::failbit);
            return is;
        }
    }
}
