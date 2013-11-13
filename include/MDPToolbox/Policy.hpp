#ifndef MDP_TOOLBOX_POLICY_HEADER_FILE
#define MDP_TOOLBOX_POLICY_HEADER_FILE

#include <cstddef>
#include <vector>
#include <tuple>

#include <ostream>

namespace MDPToolbox {
    class Policy {
        public:
            Policy(size_t sNum, size_t aNum);

            using StatePolicy = std::vector<double>;

            StatePolicy getStatePolicy(size_t s) const;

            void setPolicy(size_t s, const StatePolicy &);
            void setPolicy(size_t s, size_t a);

            size_t getS() const;
            size_t getA() const;

        private:
            size_t S, A;
            std::vector<StatePolicy> policy_;
    };

    std::ostream& operator<<(std::ostream &os, const Policy &);
}

#endif
