#ifndef MDP_TOOLBOX_POLICY_HEADER_FILE
#define MDP_TOOLBOX_POLICY_HEADER_FILE

#include <cstddef>
#include <vector>
#include <tuple>
#include <random>

#include <boost/multi_array.hpp>

namespace MDPToolbox {
    class Policy {
        public:
            using PolicyTable = boost::multi_array<double,2>;

            Policy(size_t sNum, size_t aNum);

            std::vector<double> getStatePolicy(size_t s) const;

            size_t getAction(size_t s, double epsilon = 0.0) const;

            template <typename T>
            void setPolicy(size_t s, const T &);
            void setPolicy(size_t s, size_t a);

            size_t getS() const;
            size_t getA() const;
        private:
            size_t S, A;
            PolicyTable policy_;

            // These are mutable because sampling doesn't really change the MDP
            mutable std::default_random_engine rand_;
            mutable std::uniform_real_distribution<double> sampleDistribution_;
            mutable std::uniform_int_distribution<int> randomDistribution_;
    };

    std::ostream& operator<<(std::ostream &os, const Policy &);

    template <typename T>
    void Policy::setPolicy(size_t s, const T & apt) {
        if ( std::accumulate(std::begin(apt), std::end(apt), 0.0) != 1.0 )
            throw std::runtime_error("Policy values for a state must sum to one");

        for ( size_t a = 0; a < A; a++ )
            policy_[s][a] = apt[a];
    }

}

#endif
