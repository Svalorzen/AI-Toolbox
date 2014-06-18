#ifndef AI_TOOLBOX_MDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_MDP_TYPES_HEADER_FILE

#include <vector>
#include <AIToolbox/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        // TODO: Port these to uBLAS.
        /**
         * @defgroup MDPVF MDP Value Functions.
         *
         * QFunctions and ValueFunctions are specific functions that are
         * defined in terms of policies; as in, in any particular state,
         * they can evaluate the performance that the policy will have.
         * In general however here we do not specifically specify what the
         * policy is, and since we are most probably interested in the best
         * possible policy, we try to store as little information as
         * possible in order to find that out.
         *
         * A QFunction is a function that takes in a state and action, and
         * returns the value for that particular pair. The higher the value
         * is, the better we predict we will perform. Using a QFunction to
         * obtain the perfect policy is straightforward, since at each state
         * we can simply check which action will yeld the best value, and
         * choose that one (assuming that all actions taken from that point
         * are optimal, which we would like to assume since we are trying
         * to find out the best).
         *
         * In theory, a ValueFunction is a function that is a max over
         * actions of the QFunction, as in it takes a state and returns
         * the best value obtainable from that state (following the implied
         * policy). However, that is not very useful in a practical scenario.
         * Thus we want to store not only that value, but also the action
         * that resulted in that particular choice. Instead of storing, as
         * it would make more intuitive sense, this function as a vector of
         * tuples, we are going to store it as a tuple of vectors, to allow
         * for easy manipulations of the underlying values (sums, products
         * and so on).
         *
         */

        using Values            = std::vector<double>;
        using Actions           = std::vector<size_t>;
        using ValueFunction     = std::tuple<Values, Actions>;
        enum {
            VALUES = 0,
            ACTIONS = 1,
        };
        using QFunction         = Table2D;

        /** @}  */

        /**
         * @brief This struct represents the required interface for a generative MDP.
         *
         * This struct is used to check interfaces of classes in templates.
         * In particular, this struct tests for the interface of a generative MDP model.
         * The interface must be implemented and be public in the parameter
         * class. The interface is the following:
         *
         * - size_t getA() const : Returns the number of actions of the Model.
         * - double getDiscount() const : Returns the discount factor of the Model.
         * - std::tuple<size_t, double> sampleSR(size_t s, size_t a) const : Returns a sampled state-reward pair from (s,a)
         * - bool isTerminal(size_t s) const : Reports whether the input state is a terminal state.
         *
         * is_generavie_model<M>::value will be equal to true is M implements the interface,
         * and false otherwise.
         *
         * @tparam M The class to test for the interface.
         */
        template <typename M>
        struct is_generative_model {
            private:
                template <typename Z> static auto test(int) -> decltype(

                        static_cast<size_t (Z::*)() const>                                      (&Z::getA),
                        static_cast<double (Z::*)() const>                                      (&Z::getDiscount),
                        static_cast<std::tuple<size_t, double> (Z::*)(size_t,size_t) const>     (&Z::sampleSR),
                        static_cast<bool (Z::*)(size_t) const>                                  (&Z::isTerminal),

                        std::true_type()
                );

                template <typename Z> static auto test(...) -> std::false_type;

            public:
                enum { value = std::is_same<decltype(test<M>(0)),std::true_type>::value };
        };

        /**
         * @brief This struct represents the required interface for a full MDP.
         *
         * This struct is used to check interfaces of classes in templates.
         * In particular, this struct tests for the interface of an MDP model.
         * The interface must be implemented and be public in the parameter
         * class. The interface is the following:
         *
         * - size_t getS() const : Returns the number of states of the Model.
         * - double getTransitionProbability(size_t s, size_t a, size_t s1) : Returns the transition probability given (s,a) to s1
         * - double getExpectedReward(size_t s, size_t a, size_t s1) : Returns the expected reward for transition (s,a) to s1
         *
         * In addition the MDP needs to respect the interface for the MDP generative model.
         *
         * \sa MDP::is_generative_model
         *
         * is_model<M>::value will be equal to true is M implements the interface,
         * and false otherwise.
         *
         * @tparam M The class to test for the interface.
         */
        template <typename M>
        struct is_model {
            private:
                template <typename Z> static auto test(int) -> decltype(

                        static_cast<size_t (Z::*)() const>                      (&Z::getS),
                        static_cast<double (Z::*)(size_t,size_t,size_t) const>  (&Z::getTransitionProbability),
                        static_cast<double (Z::*)(size_t,size_t,size_t) const>  (&Z::getExpectedReward),

                        std::true_type()
                );

                template <typename> static auto test(...) -> std::false_type;

            public:
                enum { value = std::is_same<decltype(test<M>(0)),std::true_type>::value && is_generative_model<M>::value };
        };
    }
}

#endif
