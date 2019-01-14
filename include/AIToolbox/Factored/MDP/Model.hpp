#ifndef AI_TOOLBOX_FACTORED_MDP_MODEL_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_MODEL_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace AIToolbox::Factored::MDP {
    class Model {
        public:
            double getTransitionProbability(const State & s, size_t a, const State & s1) const;

            double getExpectedReward(const State & s, size_t a, const State & s1) const;

        private:
            // In single-agent models, the action is still kept as a single number.
            State S; size_t A;
            double discount_;

            CompactDDN transitions_;
            std::vector<FactoredVector> rewards_;
    };
}

#endif
