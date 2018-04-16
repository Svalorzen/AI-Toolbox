#include <AIToolbox/POMDP/IO.hpp>

#include <AIToolbox/POMDP/Utils.hpp>

#include <AIToolbox/Impl/CassandraParser.hpp>

namespace AIToolbox::POMDP {
    Model<MDP::Model> parseCassandra(std::istream & input) {
        Impl::CassandraParser parser;

        const auto & [S, A, O, T, R, W, discount] = parser.parsePOMDP(input);

        return Model<MDP::Model>(O, W, S, A, T, R, discount);
    }

    std::ostream& operator<<(std::ostream &os, const Policy & p) {
        const auto & vf = p.getValueFunction();

        // VLists
        for ( size_t h = 1; h < vf.size(); ++h ) {
            const auto & vl = vf[h];
            // VEntries
            for ( auto & vv : vl ) {
                // Values
                os << vv.values.transpose() << ' ';
                // Action
                os << vv.action << ' ';
                // Obs
                for ( const auto & o : vv.observations )
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
        auto vf = makeValueFunction(S);

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
                goto failure;
            }
            // Obs
            for ( auto & o : obs ) {
                if ( !(is >> o) || ( o >= oldH && oldH ) ) {
                    goto failure;
                }
            }

            vf.back().emplace_back(std::move(values), action, std::move(obs));

            // Check if next char after whitespace is a @ that
            // marks a new horizon.
            if ( checkRemoveAtSign(is) )
                newHorizon = true;
        }

        p.H = vf.size() - 1;
        p.policy_ = std::move(vf);
        return is;

failure:
        is.setstate(std::ios::failbit);
        return is;
    }
}
