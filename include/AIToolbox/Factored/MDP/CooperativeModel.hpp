#ifndef AI_TOOLBOX_FACTORED_COOPERATIVE_MDP_MODEL_HEADER_FILE
#define AI_TOOLBOX_FACTORED_COOPERATIVE_MDP_MODEL_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace AIToolbox::Factored::MDP {
    class CooperativeModel {
        public:
            const State & getS() const;
            const Action & getA() const;
            double getDiscount() const;
            const FactoredDDN & getTransitionFunction() const;
            const Factored2DMatrix & getRewardFunction() const;
        private:
            State S;
            Action A;
            double discount_;

            FactoredDDN transitions_;
            Factored2DMatrix rewards_;
    };
}

#endif
