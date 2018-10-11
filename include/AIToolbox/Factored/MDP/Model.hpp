#ifndef AI_TOOLBOX_FACTORED_MDP_MODEL_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_MODEL_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>

namespace AIToolbox::Factored::MDP {
    class Model {
        public:

            double getTransitionProbability(const State & s, size_t a, const State & s1) const;

            double getExpectedReward(const State & s, size_t a, const State & s1) const;


        private:
            struct FactoredMatrix {
                Factors factors;
                Matrix2D matrix;
            };
            using Factored2DMatrix = std::vector<FactoredMatrix>;
            using Factored3DMatrix = boost::multi_array<FactoredMatrix, 2>;

            // In single-agent models, the action is still kept as a single number.
            State S; size_t A;
            double discount_;

            Factored3DMatrix transitions_;
            Factored2DMatrix rewards_;
    };
}

#endif
