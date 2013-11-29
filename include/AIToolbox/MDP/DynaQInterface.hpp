#ifndef AI_TOOLBOX_MDP_DYNAQINTERFACE_HEADER_FILE
#define AI_TOOLBOX_MDP_DYNAQINTERFACE_HEADER_FILE

#include <AIToolbox/MDP/QLearning.hpp>

namespace AIToolbox {
    namespace MDP {
        class DynaQInterface : public QLearning {
            public:
                DynaQInterface(double discount = 0.9, unsigned n = 50);

                virtual void updateSamplingQueue(size_t s, size_t s1, size_t a, double rew) = 0;

                virtual bool operator()(Experience &, RLModel &, Solution & s);
            private:
                double discount_;
                unsigned N_;
        };
    }
}
#endif
