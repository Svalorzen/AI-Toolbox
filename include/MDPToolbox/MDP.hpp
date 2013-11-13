#ifndef MDP_TOOLBOX_MDP_HEADER_FILE
#define MDP_TOOLBOX_MDP_HEADER_FILE

#include <cstddef>
#include <vector>
#include <tuple>

namespace MDPToolbox {
    class Policy;

    class MDP {
        public:
            using TransitionTable   = std::vector<std::vector<std::vector<double>>>;
            using RewardTable       = std::vector<std::vector<std::vector<double>>>;
            using ValueFunction     = std::vector<double>;

            MDP(size_t sNum, size_t aNum);

            template <typename T>
            void setTransitions(const T & transitions);

            template <typename T>
            void setRewards(const T & rewards);

            template <typename T>
            void setMDP(const T & mdp);

            Policy valueIteration(bool * doneOut = nullptr, double discount = 0.9, double epsilon = 0.01, unsigned maxIter = 0, ValueFunction v1 = ValueFunction(0) ) const;

            size_t getS() const;
            size_t getA() const;
        private:
            using PRType = std::vector<std::vector<double>>;
            using QType = std::vector<std::vector<double>>;

            size_t S, A;

            template <typename T>
            void setTransitions(const T & transitions, bool computePR = true);

            template <typename T>
            void setRewards(const T & rewards, bool computePR = true);

            TransitionTable transitions_;
            RewardTable rewards_;

            PRType pr_;

            std::tuple<ValueFunction, Policy> bellmanOperator(double discount, const ValueFunction & v0) const;
            void computePR();

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
                    transitions_[s][s1][a] = transitions.at(s).at(s1).at(a);
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
                    rewards_[s][s1][a] = rewards.at(s).at(s1).at(a);
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
