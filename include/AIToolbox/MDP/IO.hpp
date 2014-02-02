#ifndef AI_TOOLBOX_MDP_IO_HEADER_FILE
#define AI_TOOLBOX_MDP_IO_HEADER_FILE

#include <iostream>
#include <iomanip>

#include <type_traits>

namespace AIToolbox {
    namespace MDP {
        template <typename M, 
                  typename std::enable_if<
                                std::is_class<M>::value &&
                                std::is_member_function_pointer<decltype(&M::getS)>::value &&
                                std::is_member_function_pointer<decltype(&M::getA)>::value &&
                                std::is_member_function_pointer<decltype(&M::getTransitionProbability)>::value &&
                                std::is_member_function_pointer<decltype(&M::getExpectedReward)>::value
                            >::type* = nullptr>
        std::ostream& operator<<(std::ostream &os, const M & model) {
            size_t S = model.getS();
            size_t A = model.getA();

            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    for ( size_t a = 0; a < A; ++a ) {
                        // The +2 is for first digit and the dot, since we know that here the max value possible is 1.0
                        os << std::setw(os.precision()+2) << std::left << model.getTransitionProbability(s, s1, a) << '\t' << model.getExpectedReward(s, s1, a) << '\t';
                    }
                }
                os << '\n';
            }

            return os;
        }
    }
}
#endif
