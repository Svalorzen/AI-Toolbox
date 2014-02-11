#ifndef AI_TOOLBOX_MDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_MDP_TYPES_HEADER_FILE

#include <vector>
#include <AIToolbox/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        using ValueFunction     = std::vector<double>;
        using QFunction         = Table2D;

        template <typename T>
        struct is_model {
            private:
                template<typename U, U> struct helper{};

                template <typename Z> static auto test(Z* z) -> decltype(

                        helper<size_t (Z::*)() const,                       &Z::getS>(),
                        helper<size_t (Z::*)() const,                       &Z::getA>(),
                        helper<double (Z::*)(size_t,size_t,size_t) const,   &Z::getTransitionProbability>(),
                        helper<double (Z::*)(size_t,size_t,size_t) const,   &Z::getExpectedReward>(),

                        std::true_type());
                template <typename> static auto test(...) -> std::false_type;

            public:
                enum { value = std::is_same<decltype(test<T>((T*)nullptr)),std::true_type>::value };
        };
    }
}

#endif
