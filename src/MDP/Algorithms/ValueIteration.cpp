#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

namespace AIToolbox {
    namespace MDP {
        ValueIteration::ValueIteration(unsigned horizon, double epsilon, ValueFunction v) : horizon_(horizon), vParameter_(v),
                                                                                            S(0), A(0)
        {
            setEpsilon(epsilon);
        }

        void ValueIteration::setEpsilon(double e) {
            if ( e <= 0.0 ) throw std::invalid_argument("Epsilon must be > 0");
            epsilon_ = e;
        }
        void ValueIteration::setHorizon(unsigned h) {
            horizon_ = h;
        }
        void ValueIteration::setValueFunction(ValueFunction v) {
            vParameter_ = v;
        }

        double                  ValueIteration::getEpsilon() const {
            return epsilon_;
        }
        unsigned                ValueIteration::getHorizon() const {
            return horizon_;
        }
        const ValueFunction &   ValueIteration::getValueFunction() const {
            return vParameter_;
        }
    }
}
