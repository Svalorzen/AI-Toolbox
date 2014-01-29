#include <AIToolbox/MDP/ValueIteration.hpp>

namespace AIToolbox {
    namespace MDP {
        ValueIteration::ValueIteration(double discount, double epsilon, unsigned maxIter, ValueFunction v) : discount_(discount), epsilon_(epsilon), maxIter_(maxIter), vParameter_(v),
                                                                                                             S(0), A(0)
        {
            if ( discount_ <= 0.0 || discount_ > 1.0 )  throw std::invalid_argument("Discount parameter must be in (0,1]");
            if ( epsilon_ <= 0.0 )                      throw std::invalid_argument("Epsilon must be > 0");
        }

        void ValueIteration::setDiscount(double d) {
            if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
            discount_ = d;
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

        double                  ValueIteration::getDiscount() const {
            return discount_;
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
