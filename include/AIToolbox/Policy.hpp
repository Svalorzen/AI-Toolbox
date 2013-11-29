#ifndef AI_TOOLBOX_POLICY_HEADER_FILE
#define AI_TOOLBOX_POLICY_HEADER_FILE

#include <cstddef>
#include <vector>
#include <tuple>
#include <random>

#include <boost/multi_array.hpp>
#include <AIToolbox/Types.hpp>

namespace AIToolbox {
    class Policy {
        public:
            using PolicyTable = Table2D;

            Policy(size_t s, size_t a);

            std::vector<double> getStatePolicy(size_t s) const;

            size_t getAction(size_t s, double epsilon = 0.0) const;

            template <typename T>
            void setPolicy(size_t s, const T &);

            void setPolicy(size_t s, size_t a);

            const PolicyTable & getPolicy() const;

            size_t getS() const;
            size_t getA() const;

            void prettyPrint(std::ostream & os) const;
        private:
            size_t S, A;
            PolicyTable policy_;

            // These are mutable because sampling doesn't really change the MDP
            mutable std::default_random_engine rand_;
            mutable std::uniform_real_distribution<double> sampleDistribution_;
            mutable std::uniform_int_distribution<int> randomDistribution_;

            friend std::istream& operator>>(std::istream &is, Policy &);
    };

    std::ostream& operator<<(std::ostream &os, const Policy &);
    std::istream& operator>>(std::istream &is, Policy &);

    template <typename T>
    void Policy::setPolicy(size_t s, const T & apt) {
        double norm = static_cast<double>(std::accumulate(std::begin(apt), std::end(apt), 0.0));

        for ( size_t a = 0; a < A; a++ )
            policy_[s][a] = apt[a] / norm;
    }

}

#endif
