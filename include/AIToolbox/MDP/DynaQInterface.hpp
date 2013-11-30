#ifndef AI_TOOLBOX_MDP_DYNAQINTERFACE_HEADER_FILE
#define AI_TOOLBOX_MDP_DYNAQINTERFACE_HEADER_FILE

#include <AIToolbox/MDP/QLearning.hpp>
#include <AIToolbox/MDP/RLModel.hpp>

namespace AIToolbox {
    namespace MDP {
        class DynaQInterface : public QLearning {
            public:
                DynaQInterface(double discount = 0.9, unsigned n = 50);

                virtual void updateSamplingQueue(size_t s, size_t s1, size_t a, double rew) = 0;

                virtual void batchUpdateQ(const RLModel & m, QFunction * q) = 0;
            private:
                unsigned N_;
        };
    }
}
#endif
