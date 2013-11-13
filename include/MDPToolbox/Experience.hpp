#ifndef MDP_TOOLBOX_EXPERIENCE_HEADER_FILE
#define MDP_TOOLBOX_EXPERIENCE_HEADER_FILE

#include <string>
#include <vector>
#include <tuple>

#include <MDPToolbox/MDP.hpp>

namespace MDPToolbox {
    class Experience {
        public:
            using VisitsTable = boost::multi_array<long,3>;
            using RewardTable = MDPToolbox::MDP::Table3D;

            Experience(size_t S, size_t A);

            bool isValid() const;

            bool load(const std::string & filename);
            bool save(std::string filename = "") const;
            void update(size_t s, size_t s1, size_t a, double rew);

            std::tuple<MDPToolbox::MDP::TransitionTable, MDPToolbox::MDP::RewardTable> getMDP() const;
        private:
            bool isValid_;

            size_t S, A;

            VisitsTable visits_;
            RewardTable rewards_;
    };
}

#endif
