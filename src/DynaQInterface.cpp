#include <AIToolbox/MDP/DynaQInterface.hpp>

namespace AIToolbox {
    namespace MDP {
        DynaQInterface::DynaQInterface(size_t s, size_t a, double alpha, double discount, unsigned n) : QLearning(alpha, discount), S(s), A(a), N(n) {}

        void DynaQInterface::setN(unsigned n) {
            N = n;
        }

        unsigned DynaQInterface::getN() const {
            return N;
        }
    }
}
