#ifndef AI_TOOLBOX_MDP_QLEARNING_HEADER_FILE
#define AI_TOOLBOX_MDP_QLEARNING_HEADER_FILE

#include <stddef.h>
#include <AIToolbox/MDP/RLSolver.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        class QLearning : public RLSolver {
            public:
                QLearning(double discount = 0.9);

                void updateQ(size_t s, size_t s1, size_t a, double rew, QFunction & q);

                virtual bool operator()(Experience &, RLModel &, Solution & s);
            private:
                double discount_;
        };
    }
}
#endif
