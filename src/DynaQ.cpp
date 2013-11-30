#include <AIToolbox/MDP/DynaQ.hpp>

namespace AIToolbox {
    namespace MDP {
        DynaQ::DynaQ(double discount, unsigned n) : DynaQInterface(discount, n) {}

        /* bool DynaQ::operator()(Experience &, RLModel & model, Solution & s) {
            auto & q = s.getQFunction();
            for ( unsigned i = 0; i < N_; i++ ) {

                auto data = model.sample(4,1);
                updateQ(4, std::get<0>(data), 1, std::get<1>(data), q);
            }
        } */
    }
}
