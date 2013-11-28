#include <AIToolbox/MDP/Solution.hpp>

namespace AIToolbox {
    namespace MDP {
        Solution::Solution(size_t s, size_t a) : S(s), A(a), q_(boost::extents[S][A]), v_(S, 0.0), policy_(S, A) {}

        void Solution::setPolicy( Policy p ) {
            policy_ = p;
        }

        void Solution::setValueFunction( ValueFunction v ) {
            v_ = v;
        }

        void Solution::setQFunction( QFunction q ) {
            q_ = q;
        }

        const QFunction & Solution::getQFunction() const {
            return q_;
        }

        const ValueFunction & Solution::getValueFunction() const {
            return v_;
        }

        const Policy & Solution::getPolicy() const {
            return policy_;
        }
    }
}
