#ifndef AI_TOOLBOX_BANDIT_ROLLING_AVERAGE_HEADER_FILE
#define AI_TOOLBOX_BANDIT_ROLLING_AVERAGE_HEADER_FILE

#include <AIToolbox/Bandit/Types.hpp>

namespace AIToolbox::Bandit {
    class RollingAverage {
        public:
            RollingAverage(size_t A);

            void stepUpdateQ(size_t a, double rew);

            size_t getA() const;

            const QFunction & getQFunction() const;
            const std::vector<unsigned> & getCounts() const;

        private:
            QFunction q_;
            std::vector<unsigned> counts_;
    };
}

#endif
