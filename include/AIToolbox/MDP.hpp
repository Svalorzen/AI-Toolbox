#ifndef AI_TOOLBOX_MDP_HEADER_FILE
#define AI_TOOLBOX_MDP_HEADER_FILE

#include <vector>
#include <tuple>
#include <random>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Experience.hpp>
#include <AIToolbox/Model.hpp>
#include <AIToolbox/Policy.hpp>

namespace AIToolbox {
    class MDP {
        public:
            using ValueFunction     = std::vector<double>;
            using QFunction         = Table2D;

            MDP(size_t s, size_t a);
            MDP(Model);

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

            Model & getModel();
            const Model & getModel() const;
        private:
            using PRType = Table2D;

            size_t S, A;

            Model model_;

            bool prValid_;
            PRType pr_;

            QFunction q_;
            ValueFunction v_;
            Policy policy_;

            // These are mutable because sampling doesn't really change the MDP
            mutable std::default_random_engine rand_;
            mutable std::uniform_real_distribution<double> sampleDistribution_;

            void computePR();
            void updateQ(size_t s, size_t s1, size_t a, double rew, double discount);

            std::tuple<QFunction, ValueFunction, Policy> bellmanOperator(double discount, const ValueFunction & v0) const;
            unsigned valueIterationBoundIter(double discount, double epsilon, const ValueFunction & v0) const;
    };

}

#endif
