#ifndef AI_TOOLBOX_MDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_MDP_TYPES_HEADER_FILE

#include <vector>
#include <AIToolbox/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        using ValueFunction     = std::vector<double>;
        using QFunction         = Table2D;

        /**
         * @brief This struct represents the required interface for a Model.
         *
         * This struct is used to check interfaces of classes in templates.
         * In particular, this struct tests for the interface of an MDP model.
         * The interface must be implemented and be public in the parameter
         * class. The interface is the following:
         *
         * - size_t getS() const : Returns the number of states of the Model.
         * - size_t getA() const : Returns the number of actions of the Model.
         * - double getTransitionProbability(size_t s, size_t a, size_t s1) : Returns the transition probability given (s,a) to s1
         * - double getExpectedReward(size_t s, size_t a, size_t s1) : Returns the expected reward for transition (s,a) to s1
         *
         * is_model<M>::value will be equal to true is M implements the interface,
         * and false otherwise.
         *
         * @tparam T The class to test for the interface.
         */
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
