#ifndef TABLE_HEADER_FILE
#define TABLE_HEADER_FILE

#include <string>
#include <vector>
#include <tuple>

#include <MDPToolbox/MDP.hpp>

class Table {
    public:
        // Since we have to actions, we record how many times we did one or the other in a particular MDPState-MDPState transition.
        // In addition we also record the total rewards obtained in doing so. A1, A2, A1rew, A2rew.
        typedef std::tuple<long, double> TransitionVisitsRewards;
        enum {
            visits = 0,
            reward = 1
        };

        typedef std::vector<TransitionVisitsRewards> EntryType;
        typedef std::vector<EntryType> TransitionType;
        typedef std::vector<TransitionType> TableType;

        Table(size_t S, size_t A);

        bool isValid();

        bool load(const std::string & filename);
        bool save(std::string filename = "");

        std::tuple<MDPToolbox::MDP::TransitionTable, MDPToolbox::MDP::RewardTable> getMDP() const;
    private:
        std::string lastFilename_;
        bool isValid_;

        size_t S, A;

        TableType table_;
};

#endif
