#ifndef AI_TOOLBOX_POMDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_POMDP_TYPES_HEADER_FILE

#include <vector>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace POMDP {
        using Belief            = std::vector<double>;

        /**
         * @brief This struct represents the required interface for a POMDP Model.
         *
         * This struct is used to check interfaces of classes in templates.
         * In particular, this struct tests for the interface of a POMDP model.
         * The interface must be implemented and be public in the parameter
         * class. The interface is the following:
         *
         * - size_t getO() const : Returns the number of observations of the Model.
         * - double getObservationProbability(size_t s1, size_t a, size_t o) : Returns the probability for observation o after action a and final state s1.
         *
         * In addition the POMDP needs to respect the interface for the MDP Model.
         *
         * \sa MDP::is_model
         *
         * is_model<M>::value will be equal to true is M implements the interface,
         * and false otherwise.
         *
         * @tparam T The class to test for the interface.
         */
        template <typename T>
        struct is_model {
            private:
                template <typename Z> static auto test(int) -> decltype(

                        static_cast<size_t (Z::*)() const>                       (&Z::getO),
                        static_cast<double (Z::*)(size_t,size_t,size_t) const>   (&Z::getObservationProbability),

                        std::true_type()
                );

                template <typename> static auto test(...) -> std::false_type;

            public:
                enum { value = std::is_same<decltype(test<T>(0)),std::true_type>::value && MDP::is_model<T>::value };
        };
    }
}

#endif
