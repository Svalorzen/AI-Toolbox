#ifndef MDP_TOOLBOX_POLICY_HEADER_FILE
#define MDP_TOOLBOX_POLICY_HEADER_FILE

#include <cstddef>
#include <vector>
#include <tuple>

namespace MDPToolbox {
    class Policy {
        public:
            Policy(size_t sNum, size_t aNum);

            using ActionPolicyType = std::vector<double>;

            ActionPolicyType getProbability(size_t s) const;

            void setPolicy(size_t s, const ActionPolicyType &);
            void setPolicy(size_t s, size_t a);

            size_t getS() const;
            size_t getA() const;
        private:
            size_t S, A;
            std::vector<ActionPolicyType> policy_; 
    };
}

#endif
