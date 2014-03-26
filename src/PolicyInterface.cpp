#include <AIToolbox/PolicyInterface.hpp>

#include <ostream>


namespace AIToolbox {
    std::ostream& operator<<(std::ostream &os, const PolicyInterface<size_t> &p) {
        size_t S = p.getS();
        size_t A = p.getA();

        for ( size_t s = 0; s < S; ++s ) {
            for ( size_t a = 0; a < A; ++a ) {
                os << s << "\t" << a << "\t" << std::fixed << p.getActionProbability(s,a) << "\n";
            }
        }
        return os;
    }
}
