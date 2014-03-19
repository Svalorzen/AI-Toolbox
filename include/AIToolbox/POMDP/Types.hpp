#ifndef AI_TOOLBOX_POMDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_POMDP_TYPES_HEADER_FILE

#include <vector>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace POMDP {
        using Belief            = std::vector<double>;

        /**
         * @brief This struct represents the required interface for a Model.
         *
         * This struct is used to check interfaces of classes in templates.
         * In particular, this struct tests for the interface of a POMDP model.
         * The interface must be implemented and be public in the parameter
         * class. The interface is the following:
         *
         * - size_t getO() const : Returns the number of observations of the Model.
         * - const MDP & getMDP() const : Returns a const reference to a type which satisfies MDP::is_model::value == true.
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

                template <typename R, typename C, typename... Args>
                static R get_return_type (R (C::*)(Args...));

                template <typename Z> static auto test(Z* z) -> decltype(

                        helper<size_t                                 (Z::*)() const,                       &Z::getO>(),

                        // For getMDP we need a two phase check, since we also need to assure that the returned
                        // value is an actual MDP. For this method we want a const reference, so we will enforce this.
                        helper<decltype(get_return_type(&Z::getMDP))  (Z::*)() const,                       &Z::getMDP>(),

                        typename std::enable_if<

                                std::is_reference<                               decltype(get_return_type(&Z::getMDP))>       ::value   &&
                                std::is_const    <typename std::remove_reference<decltype(get_return_type(&Z::getMDP))>::type>::value   &&
                                MDP::is_model    <typename std::remove_reference<decltype(get_return_type(&Z::getMDP))>::type>::value

                        ,int>::type(),

                        std::true_type());
                template <typename> static auto test(...) -> std::false_type;

            public:
                enum { value = std::is_same<decltype(test<T>((T*)nullptr)),std::true_type>::value };
        };
    }
}

#endif
