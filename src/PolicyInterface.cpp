#include <AIToolbox/PolicyInterface.hpp>

#include <ostream>

#include "Seeder.hpp"

namespace AIToolbox {
    PolicyInterface::PolicyInterface(size_t s, size_t a) : S(s), A(a),
                                                           rand_(Impl::Seeder::getSeed()),
                                                           sampleDistribution_(0.0, 1.0) {}

    PolicyInterface::~PolicyInterface() {}

    size_t PolicyInterface::getS() const { return S; }
    size_t PolicyInterface::getA() const { return A; }

    std::ostream& operator<<(std::ostream &os, const PolicyInterface &p) {
        size_t S = p.getS();
        size_t A = p.getA();

        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t a = 0; a < A; a++ ) {
                os << s << "\t" << a << "\t" << std::fixed << p.getActionProbability(s,a) << "\n";
            }
        }
        return os;
    }
}
