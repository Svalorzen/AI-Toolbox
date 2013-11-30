#ifndef AI_TOOLBOX_MDP_PRIORITIZEDSWEEPING_HEADER_FILE
#define AI_TOOLBOX_MDP_PRIORITIZEDSWEEPING_HEADER_FILE

#include <AIToolbox/MDP/DynaQInterface.hpp>

namespace AIToolbox {
    namespace MDP {
        class PrioritizedSweeping : public DynaQInterface {
            public:
                PrioritizedSweeping(double discount = 0.9, unsigned n = 50);

                virtual void updateSamplingQueue(size_t s, size_t s1, size_t a, double rew);

                virtual void batchUpdateQ(const RLModel & m, QFunction * q) = 0;
            private:
        };
    }
}
#endif
