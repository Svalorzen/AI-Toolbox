#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/MDP/Policies/Policy.hpp>

namespace AIToolbox::MDP {
    QFunction makeQFunction(const size_t S, const size_t A) {
        auto retval = QFunction(S, A);
        retval.setZero();
        return retval;
    }

    ValueFunction makeValueFunction(const size_t S) {
        auto values = Values(S);
        values.setZero();
        return {values, Actions(S, 0)};
    }

    ValueFunction bellmanOperator(const QFunction & q) {
        const auto S = q.rows();
        ValueFunction vf{Values(S), Actions(S)};
        bellmanOperatorInplace(q, &vf);
        return vf;
    }

    void bellmanOperatorInplace(const QFunction & q, ValueFunction * v) {
        assert(v);
        auto & values  = v->values;
        auto & actions = v->actions;

        for ( size_t s = 0; s < actions.size(); ++s )
            values(s) = q.row(s).maxCoeff(&actions[s]);
    }
}
