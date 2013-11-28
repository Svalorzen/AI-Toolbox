#ifndef AI_TOOLBOX_MDP_SOLUTION_HEADER_FILE
#define AI_TOOLBOX_MDP_SOLUTION_HEADER_FILE

#include <AIToolbox/Policy.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        class Solution {
            public:
                void setPolicy(Policy p);
                void setValueFunction(ValueFunction v);
                void setQFunction(QFunction q);

                const Policy &          getPolicy()             const;
                const ValueFunction &   getValueFunction()      const;
                const QFunction &       getQFunction()          const;
            private:
                QFunction q_;
                ValueFunction v_;
                Policy policy_;
        };
    }
}

#endif
