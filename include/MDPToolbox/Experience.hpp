#ifndef MDP_TOOLBOX_EXPERIENCE_HEADER_FILE
#define MDP_TOOLBOX_EXPERIENCE_HEADER_FILE

#include <string>
#include <vector>
#include <tuple>

#include <boost/multi_array.hpp>

namespace MDPToolbox {
    class Experience {
        public:
            using VisitTable = boost::multi_array<unsigned long,3>;
            using RewardTable = boost::multi_array<double,3>;

            Experience(size_t S, size_t A);

            void update(size_t s, size_t s1, size_t a, double rew);
            void reset();

            const VisitTable  & getVisits() const;
            const RewardTable & getRewards() const;

            size_t getS() const;
            size_t getA() const;
        private:
            size_t S, A;

            VisitTable visits_;
            RewardTable rewards_;

            friend std::istream& operator>>(std::istream &os, Experience &);
    };

    std::ostream& operator<<(std::ostream &os, const Experience &);
    std::istream& operator>>(std::istream &is, Experience &);
}

#endif
