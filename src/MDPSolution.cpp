#include <AIToolbox/MDP/Solution.hpp>

namespace AIToolbox {
    namespace MDP {
        Solution::Solution(size_t s, size_t a) : S(s), A(a), q_(boost::extents[S][A]), v_(S, 0.0), policy_(S, A) {
            std::fill(q_.data(), q_.data() + q_.num_elements(), 0.0);
        }

        Policy &          Solution::getPolicy()                     { return policy_;   }
        const Policy & Solution::getPolicy()                const   { return policy_;   }

        ValueFunction &   Solution::getValueFunction()              { return v_;        }
        const ValueFunction & Solution::getValueFunction()  const   { return v_;        }

        QFunction &       Solution::getQFunction()                  { return q_;        }
        const QFunction & Solution::getQFunction()          const   { return q_;        }

        size_t Solution::getGreedyQAction(size_t s) const {
            return std::distance(std::begin(q_[s]), std::max_element(std::begin(q_[s]), std::end(q_[s])));
        }
    }
}
