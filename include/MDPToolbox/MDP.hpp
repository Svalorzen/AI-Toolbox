#ifndef MDP_TOOLBOX_MDP_HEADER_FILE
#define MDP_TOOLBOX_MDP_HEADER_FILE

#include <cstddef>
#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <functional>

#include <boost/multi_array.hpp>
#include <MDPToolbox/Experience.hpp>
#include <MDPToolbox/Policy.hpp>

namespace MDPToolbox {
    class MDP {
        public:
            using Table3D = boost::multi_array<double, 3>;
            using Table2D = boost::multi_array<double, 2>;
            using TransitionTable   = Table3D;
            using RewardTable       = Table3D;
            using ValueFunction     = std::vector<double>;
            using QFunction         = Table2D;

            MDP(const Experience &);

            template <typename T, typename U>
            MDP(const T & transitions, const U & rewards);

            void                        updateModel(size_t s, size_t s1, size_t a, double reward);
            std::tuple<size_t, double>  sampleModel(size_t s, size_t a) const;

            void                        updatePrioritizedSweepingQueue(size_t s, size_t s1, size_t a, double reward);

            bool valueIteration(double discount = 0.9, double epsilon = 0.01, unsigned maxIter = 0, ValueFunction v1 = ValueFunction(0) );
            void DynaQ         (std::function<std::tuple<size_t, size_t>()> generator, double discount = 0.9, unsigned n = 50);

            const Policy &          getPolicy()             const;
            const ValueFunction &   getValueFunction()      const;
            const QFunction &       getQFunction()          const;
            const TransitionTable & getTransitionFunction() const;
            const RewardTable &     getRewardFunction()     const;

            size_t getGreedyAction(size_t s) const;

            size_t getS() const;
            size_t getA() const;
        private:
            using PRType = Table2D;

            size_t S, A;

            TransitionTable transitions_;
            RewardTable rewards_;

            bool prValid_;
            PRType pr_;

            QFunction q_;
            ValueFunction v_;
            Policy policy_;

            // These are mutable because sampling doesn't really change the MDP
            mutable std::default_random_engine rand_;
            mutable std::uniform_real_distribution<double> sampleDistribution_;

            template <typename T>
            void setTransitions(const T & transitions, bool computePR = true);

            template <typename T>
            void setRewards(const T & rewards, bool computePR = true);

            void computePR();
            std::tuple<QFunction, ValueFunction, Policy> bellmanOperator(double discount, const ValueFunction & v0) const;
            unsigned valueIterationBoundIter(double discount, double epsilon, const ValueFunction & v0) const;
    };

    template <typename T, typename U>
    MDP::MDP(const T & transitions, const U & rewards) :
                                         S(transitions.size()), A(transitions.at(0).at(0).size()), transitions_(boost::extents[S][S][A]), rewards_(boost::extents[S][S][A]), prValid_(false), pr_(boost::extents[S][A]),
                                         q_(boost::extents[S][A]), v_(S,0.0), policy_(S,A),
                                         rand_(std::chrono::system_clock::now().time_since_epoch().count()), sampleDistribution_(0.0, 1.0)
    {
        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                double pCheck = 0.0;
                for ( size_t a = 0; a < A; a++ ) {
                    transitions_[s][s1][a] = transitions[s][s1][a];
                    rewards_[s][s1][a] = rewards[s][s1][a];

                    pCheck += transitions_[s][s1][a];
                }
                if ( pCheck != 1.0 ) {
                    throw std::runtime_error("Input transition matrix does not contain real probabilities.");
                }
            }
        }
        computePR();
    }
}

#endif
