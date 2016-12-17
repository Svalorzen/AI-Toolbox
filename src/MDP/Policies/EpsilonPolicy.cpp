#include <AIToolbox/MDP/Policies/EpsilonPolicy.hpp>

namespace AIToolbox {
    namespace MDP {
        EpsilonPolicy::EpsilonPolicy(const Base::Base & p, double epsilon) :
                Base(p, epsilon), randomDistribution_(0, this->A-1) {}

        size_t EpsilonPolicy::sampleRandomAction() const {
            return randomDistribution_(rand_);
        }
    }
}
