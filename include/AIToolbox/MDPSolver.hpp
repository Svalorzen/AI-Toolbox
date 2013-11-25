#ifndef AI_TOOLBOX_MDPSOLVER_HEADER_FILE
#define AI_TOOLBOX_MDPSOLVER_HEADER_FILE

#include <vector>
#include <tuple>
#include <random>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Experience.hpp>
#include <AIToolbox/MDP.hpp>
#include <AIToolbox/Policy.hpp>

namespace AIToolbox {
    class MDPSolver {
        public:
            using ValueFunction     = std::vector<double>;
            using QFunction         = Table2D;

            MDPSolver(size_t s, size_t a);
            MDPSolver(MDP);

            void updatePrioritizedSweepingQueue(size_t s, double discount, double threshold);

            void                        update();
            void                        update(size_t s, size_t a);

            bool valueIteration     (double discount = 0.9, double epsilon = 0.01, unsigned maxIter = 0, ValueFunction v1 = ValueFunction(0) );
            void dynaQ              (size_t s, size_t a, double discount);
            void prioritizedSweeping(double discount, double threshold);

            const Policy &          getPolicy()             const;
            const ValueFunction &   getValueFunction()      const;
            const QFunction &       getQFunction()          const;

            size_t getGreedyAction(size_t s) const;

            size_t getS() const;
            size_t getA() const;

            MDP & getMDP();
            const MDP & getMDP() const;
        private:
            using PRType = Table2D;

            size_t S, A;

            MDP model_;

            bool prValid_;
            PRType pr_;

            QFunction q_;
            ValueFunction v_;
            Policy policy_;

            void computePR();
            void updateQ(size_t s, size_t s1, size_t a, double rew, double discount);

            std::tuple<QFunction, ValueFunction, Policy> bellmanOperator(double discount, const ValueFunction & v0) const;
            unsigned valueIterationBoundIter(double discount, double epsilon, const ValueFunction & v0) const;
    };

}

#endif
