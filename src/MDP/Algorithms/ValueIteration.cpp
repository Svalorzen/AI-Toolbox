#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

namespace AIToolbox {
    namespace MDP {
        ValueIteration::ValueIteration(double epsilon, unsigned maxIter, ValueFunction v) : maxIter_(maxIter), vParameter_(v),
                                                                                                             S(0), A(0)
        {
            setEpsilon(epsilon);
        }

        void ValueIteration::setEpsilon(double e) {
            if ( e <= 0.0 ) throw std::invalid_argument("Epsilon must be > 0");
            epsilon_ = e;
        }
        void ValueIteration::setMaxIter(unsigned m) {
            maxIter_ = m;
        }
        void ValueIteration::setValueFunction(ValueFunction v) {
            vParameter_ = v;
        }

        double                  ValueIteration::getEpsilon() const {
            return epsilon_;
        }
        unsigned                ValueIteration::getMaxIter() const {
            return maxIter_;
        }
        const ValueFunction &   ValueIteration::getValueFunction() const {
            return vParameter_;
        }
    }
}
