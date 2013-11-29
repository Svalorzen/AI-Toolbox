#include <AIToolbox/MDP/DynaQ.hpp>

#include <AIToolbox/Experience.hpp>
#include <AIToolbox/MDP/Solution.hpp>
#include <AIToolbox/MDP/RLModel.hpp>

namespace AIToolbox {
    namespace MDP {
        DynaQ::DynaQ(double discount, unsigned n) : discount_(discount), N_(n) {
            if ( discount_ <= 0 || discount_ > 1 )  throw std::runtime_error("Discount parameter must be in (0,1]");
        }

        bool DynaQ::operator()(Experience &, RLModel & model, Solution & s) {
            auto & q = s.getQFunction();
            for ( unsigned i = 0; i < N_; i++ ) {

                auto data = model.sample(4,1);
                updateQ(4, std::get<0>(data), 1, std::get<1>(data), q);
            }
        }

        void DynaQ::updateQ(size_t s, size_t s1, size_t a, double rew, QFunction & q) {
            q[s][a] += discount_ * ( rew * (*std::max_element(std::begin(q[s1]),std::end(q[s1]))) - q[s][a] );
        }

    }
}
