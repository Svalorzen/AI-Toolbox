#ifndef AI_TOOLBOX_POMDP_UTILS_TYPES_HEADER_FILE
#define AI_TOOLBOX_POMDP_UTILS_TYPES_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>

namespace AIToolbox {
    namespace POMDP {
        /**
         * @brief This check the interface for a WitnessLP.
         *
         * @tparam LP The type of the LP to be checked.
         */
        template <typename LP>
        struct is_witness_lp {
            private:
                template <typename Z> static auto test(int) -> decltype(

                        Z(0), // Check we can build it from a size_t
                        static_cast<void (Z::*)()>                                                  (&Z::reset),
                        static_cast<void (Z::*)(size_t size)>                                       (&Z::allocate),
                        static_cast<void (Z::*)(const MDP::Values &)>                               (&Z::addOptimalRow),
                        static_cast<std::tuple<bool, Belief> (Z::*)(const MDP::Values &)>           (&Z::findWitness),

                        std::true_type()
                );

                template <typename Z> static auto test(...) -> std::false_type;

            public:
                enum { value = std::is_same<decltype(test<LP>(0)),std::true_type>::value };
        };
    }
}

#endif
