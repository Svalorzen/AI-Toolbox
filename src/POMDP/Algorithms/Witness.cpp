#include <AIToolbox/POMDP/Algorithms/Witness.hpp>

namespace AIToolbox {
    namespace POMDP {
        Witness::Witness(unsigned h, double e) : horizon_(h) {
            setEpsilon(e);
        }

        void Witness::setHorizon(unsigned h) {
            horizon_ = h;
        }
        void Witness::setEpsilon(double e) {
            if ( e < 0.0 ) throw std::invalid_argument("Epsilon must be >= 0");
            epsilon_ = e;
        }

        unsigned Witness::getHorizon() const {
            return horizon_;
        }

        double Witness::getEpsilon() const {
            return epsilon_;
        }
    }
}
