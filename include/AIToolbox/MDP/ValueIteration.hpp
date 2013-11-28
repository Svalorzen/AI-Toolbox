#ifndef AI_TOOLBOX_MDP_VALUE_ITERATION_HEADER_FILE
#define AI_TOOLBOX_MDP_VALUE_ITERATION_HEADER_FILE

#include <AIToolbox/MDP/Solver.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    class Policy;
    namespace MDP {
        class ValueIteration : public Solver {
                ValueIteration(double discount = 0.9, double epsilon = 0.01, unsigned maxIter = 0, ValueFunction v = ValueFunction(0) );

                virtual bool solve(const Model & model, Solution & solution);
            private:
                using PRType = Table2D;
                
                double discount_, epsilon_;
                unsigned maxIter_;
                ValueFunction vParameter_;
                ValueFunction v1_;

                const Model * model_;
                size_t S, A;

                PRType computePR() const;

                unsigned valueIterationBoundIter(const PRType & pr) const;

                QFunction makeQFunction(const PRType & pr) const;
                void bellmanOperator(const PRType & pr, ValueFunction & vOut) const;
        };

    }
}

#endif
