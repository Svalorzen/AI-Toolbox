#ifndef MDP_TOOLBOX_MDP_HEADER_FILE
#define MDP_TOOLBOX_MDP_HEADER_FILE

#include <cstddef>
#include <vector>
#include <tuple>

namespace MDPToolbox {
    class Policy;

    class MDP {
        public:
            using MDPType = std::vector<std::vector<std::vector<std::tuple<double, double>>>>;
            using ValueType = std::vector<double>;

            MDP(size_t sNum, size_t aNum);

            template <typename T>
            void setPolicy(const T & mdp);

            Policy valueIteration(double discount, double epsilon = 0.01, unsigned maxIter = 0, ValueType v0 = ValueType(0), bool * doneOut = nullptr ) const;

            size_t getS() const;
            size_t getA() const;
        private:
            using PRType = std::vector<std::vector<double>>;
            using QType = std::vector<std::vector<double>>;

            size_t S, A;

            MDPType mdp_; 
            enum {
                Probability,
                Reward
            };
            PRType pr_;

            std::tuple<ValueType, Policy> bellmanOperator(double discount, const ValueType & v0) const;
            void computePR();

            unsigned valueIterationBoundIter(double discount, double epsilon, const ValueType & v0) const;
    };

    template <typename T>
    void MDP::setPolicy(const T & mdp) {
        for ( size_t s = 0; s < S; s++ )
            for ( size_t s1 = 0; s1 < S; s1++ )
                for ( size_t a = 0; a < A; a++ )
                    mdp_[s][s1][a] = mdp.at(s).at(s1).at(a);
    }
}

#endif
