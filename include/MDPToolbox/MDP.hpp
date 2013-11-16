#ifndef MDP_TOOLBOX_MDP_HEADER_FILE
#define MDP_TOOLBOX_MDP_HEADER_FILE

#include <cstddef>
#include <vector>
#include <tuple>
#include <random>

#include <boost/multi_array.hpp>
#include <MDPToolbox/Policy.hpp>

namespace MDPToolbox {
    class Policy;

    class MDP {
        public:
            using Table3D = boost::multi_array<double, 3>;
            using Table2D = boost::multi_array<double, 2>;
            using TransitionTable   = Table3D;
            using RewardTable       = Table3D;
            using ValueFunction     = std::vector<double>;
            using QFunction         = Table2D;

            MDP(size_t sNum, size_t aNum);

            template <typename T>
            void setTransitions(const T & transitions);

            template <typename T>
            void setRewards(const T & rewards);

            template <typename T>
            void setMDP(const T & mdp);

            std::tuple<size_t, double> sample(size_t s, size_t a) const;
            bool valueIteration(double discount = 0.9, double epsilon = 0.01, unsigned maxIter = 0, ValueFunction v1 = ValueFunction(0) );

            const Policy & getPolicy() const;
            const ValueFunction & getValueFunction() const;
            const QFunction & getQFunction() const;

            size_t getS() const;
            size_t getA() const;
        private:
            using PRType = Table2D;

            size_t S, A;

            TransitionTable transitions_;
            RewardTable rewards_;

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

    template <typename T>
    void MDP::setTransitions(const T & transitions) {
        setTransitions(transitions, true);
    }

    template <typename T>
    void MDP::setTransitions(const T & transitions, bool compute ) {
        for ( size_t s = 0; s < S; s++ )
            for ( size_t s1 = 0; s1 < S; s1++ )
                for ( size_t a = 0; a < A; a++ )
                    transitions_[s][s1][a] = transitions[s][s1][a];
        if ( compute )
            computePR();
    }

    template <typename T>
    void MDP::setRewards(const T & rewards) {
        setRewards(rewards, true);
    }

    template <typename T>
    void MDP::setRewards(const T & rewards, bool compute ) {
        for ( size_t s = 0; s < S; s++ )
            for ( size_t s1 = 0; s1 < S; s1++ )
                for ( size_t a = 0; a < A; a++ )
                    rewards_[s][s1][a] = rewards[s][s1][a];
        if ( compute )
            computePR();
    }

    template <typename T>
    void MDP::setMDP(const T & mdp) {
        setTransitions(std::get<0>(mdp), false);
        setRewards(std::get<1>(mdp), true);
    }
}

#endif
