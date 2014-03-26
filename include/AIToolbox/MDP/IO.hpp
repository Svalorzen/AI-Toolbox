#ifndef AI_TOOLBOX_MDP_IO_HEADER_FILE
#define AI_TOOLBOX_MDP_IO_HEADER_FILE

#include <iostream>
#include <iomanip>

#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {

    template<typename State> class PolicyInterface;

    namespace MDP {
        template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
        std::ostream& operator<<(std::ostream &os, const M & model) {
            size_t S = model.getS();
            size_t A = model.getA();

            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    for ( size_t a = 0; a < A; ++a ) {
                        // The +2 is for first digit and the dot, since we know that here the max value possible is 1.0
                        os << std::setw(os.precision()+2) << std::left << model.getTransitionProbability(s, s1, a) << '\t'
                           << std::setw(os.precision()+2) << std::left << model.getExpectedReward(s, s1, a)        << '\t';
                    }
                }
                os << '\n';
            }

            return os;
        }

        /**
         * @brief This function prints the whole policy to a stream.
         *
         * This function outputs each and every value of the policy
         * for easy parsing. The output is broken into multiple lines
         * where each line is of the format:
         *
         * state_number action_number probability
         *
         * And all lines are sorted by state, and each state is sorted
         * by action.
         *
         * @param os The stream where the policy is printed.
         * @param p The policy that is begin printed.
         *
         * @return The original stream.
         */
        std::ostream& operator<<(std::ostream &os, const PolicyInterface<size_t> & p);
    }
}
#endif
