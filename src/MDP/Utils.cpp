#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/MDP/Policies/Policy.hpp>

namespace AIToolbox {
    namespace MDP {
        QFunction makeQFunction(const size_t S, const size_t A) {
            auto retval = QFunction(S, A);
            retval.fill(0.0);
            return retval;
        }

        ValueFunction makeValueFunction(const size_t S) {
            auto values = Values(S);
            values.fill(0.0);
            return std::make_tuple(values, Actions(S, 0));
        }

        ValueFunction bellmanOperator(const QFunction & q) {
            const auto S = q.rows();
            auto vf = std::make_tuple(Values(S), Actions(S));
            bellmanOperatorInline(q, &vf);
            return vf;
        }

        void bellmanOperatorInline(const QFunction & q, ValueFunction * v) {
            assert(v);
            auto & values  = std::get<VALUES> (*v);
            auto & actions = std::get<ACTIONS>(*v);

            for ( size_t s = 0; s < actions.size(); ++s )
                values(s) = q.row(s).maxCoeff(&actions[s]);
        }
    }
}
