#ifndef AI_TOOLBOX_MDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_MDP_TYPES_HEADER_FILE

#include <vector>

#include <AIToolbox/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @name MDP Value Types
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
         * @{
         */

        using Values            = Vector;
        using Actions           = std::vector<size_t>;
        using ValueFunction     = std::tuple<Values, Actions>;
        enum {
            VALUES = 0,
            ACTIONS = 1,
        };

        using QFunction = Matrix2D;

        /** @}  */

        /**
         * @brief This struct represents the required interface for a generative MDP.
         *
         * This struct is used to check interfaces of classes in templates.
         * In particular, this struct tests for the interface of a generative MDP model.
         * The interface must be implemented and be public in the parameter
         * class. The interface is the following:
         *
         * - size_t getS() const : Returns the number of states of the Model.
         * - size_t getA() const : Returns the number of actions of the Model.
         * - double getDiscount() const : Returns the discount factor of the Model.
         * - std::tuple<size_t, double> sampleSR(size_t s, size_t a) const : Returns a sampled state-reward pair from (s,a)
         * - bool isTerminal(size_t s) const : Reports whether the input state is a terminal state.
         *
         * is_generative_model<M>::value will be equal to true is M implements the interface,
         * and false otherwise.
         *
         * @tparam M The class to test for the interface.
         */
        template <typename M>
        struct is_generative_model {
            private:
                template <typename Z> static constexpr auto test(int) -> decltype(

                        static_cast<size_t (Z::*)() const>                                      (&Z::getS),
                        static_cast<size_t (Z::*)() const>                                      (&Z::getA),
                        static_cast<double (Z::*)() const>                                      (&Z::getDiscount),
                        static_cast<std::tuple<size_t, double> (Z::*)(size_t,size_t) const>     (&Z::sampleSR),
                        static_cast<bool (Z::*)(size_t) const>                                  (&Z::isTerminal),

                        bool()
                ) { return true; }

                template <typename Z> static constexpr auto test(...) -> bool
                { return false; }

            public:
                enum { value = test<M>(0) };
        };

        /**
         * @brief This struct represents the required interface for a full MDP.
         *
         * This struct is used to check interfaces of classes in templates.
         * In particular, this struct tests for the interface of an MDP model.
         * The interface must be implemented and be public in the parameter
         * class. The interface is the following:
         *
         * - double getTransitionProbability(size_t s, size_t a, size_t s1) const : Returns the transition probability given (s,a) to s1
         * - double getExpectedReward(size_t s, size_t a, size_t s1) const : Returns the expected reward for transition (s,a) to s1
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
                template <typename Z> static constexpr auto test(int) -> decltype(

                        static_cast<double (Z::*)(size_t,size_t,size_t) const>  (&Z::getTransitionProbability),
                        static_cast<double (Z::*)(size_t,size_t,size_t) const>  (&Z::getExpectedReward),

                        bool()
                ) { return true; }

                template <typename> static constexpr auto test(...) -> bool
                { return false; }

            public:
                enum { value = test<M>(0) && is_generative_model<M>::value };
        };

        /**
         * @brief This struct represents the required interface that allows MDP algorithms to leverage Eigen.
         *
         * This struct is used to check interfaces of classes in templates.
         * In particular, this struct tests for the interface of an MDP model
         * which uses Eigen matrices internally.
         * The interface must be implemented and be public in the parameter
         * class. The interface is the following:
         *
         * - T getTransitionFunction(size_t a) const : Returns the transition function for a given action as a matrix SxS', where T is some Eigen matrix type.
         * - R getRewardFunction(size_t a) const : Returns the reward function for a given action as a matrix SxS', where R is some Eigen matrix type.
         *
         * In addition the MDP needs to respect the interface for the MDP model.
         *
         * \sa MDP::is_model
         *
         * is_model_eigen<M>::value will be equal to true is M implements the interface,
         * and false otherwise.
         *
         * @tparam M The class to test for the interface.
         */
        template <typename M>
        struct is_model_eigen {
            private:
                template <typename T>
                struct remove_cv_ref { using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type; };

                // With this macro we can find out the return type of a given member function; we use it
                // so that we can check whether the class offers methods which return Eigen types, so we
                // can enable the high-performance algorithm variants.
                #define RETVAL_EXTRACTOR(fun_name)                                                                                                  \
                                                                                                                                                    \
                template <typename Z, typename ...Args> static auto fun_name##RetType(Z* z) ->                                                      \
                                                                    typename remove_cv_ref<decltype(z->fun_name(std::declval<Args>()...))>::type;   \
                                                                                                                                                    \
                template <typename Z, typename ...Args> static auto fun_name##RetType(...) -> int

                RETVAL_EXTRACTOR(getTransitionFunction);
                RETVAL_EXTRACTOR(getRewardFunction);

                // The template parameters here must match the ones used in the test function!
                // So const M if the function is const, and then the parameter types.
                using F = decltype(getTransitionFunctionRetType<const M, size_t>(0));
                using R = decltype(getRewardFunctionRetType<const M, size_t>(0));

                template <typename Z> static constexpr auto test(int) -> decltype(

                        static_cast<const F & (Z::*)(size_t) const>         (&Z::getTransitionFunction),
                        static_cast<const R & (Z::*)(size_t) const>         (&Z::getRewardFunction),

                        bool()
                ) { return true; }

                template <typename Z> static constexpr auto test(...) -> bool
                { return false; }

                #undef RETVAL_EXTRACTOR

            public:
                enum {
                    value = is_model<M>::value && test<M>(0) &&
                            std::is_base_of<Eigen::EigenBase<F>, F>::value &&
                            std::is_base_of<Eigen::EigenBase<R>, R>::value
                };
        };

        /**
         * @brief This struct verifies that a class satisfies the is_model interface but not the is_model_eigen interface.
         *
         * @tparam M The class to test for the interface.
         */
        template <typename M>
        struct is_model_not_eigen {
            public:
                enum { value = is_model<M>::value && !is_model_eigen<M>::value };
        };

        /**
         * @brief This struct represents the required interface for an experience recorder.
         *
         * This struct is used to check interfaces of classes in templates.
         * In particular, this struct tests for the interface of an experience
         * recorder that can be used to create Reinforcement Learning MDP models.
         * The interface must be implemented and be public in the parameter
         * class. The interface is the following:
         *
         * - long unsigned getVisits(size_t, size_t, size_t) const : Returns the number of times a particular transition has been experienced.
         * - long unsigned getVisitsSum(size_t, size_t) const : Returns the number of times a transition starting with the parameters has been experienced.
         * - double getReward(size_t, size_t, size_t) const : Returns the cumulative rewards obtained from a specific transition.
         * - double getRewardSum(size_t, size_t) const : Returns the cumulative rewards obtained from transitions starting with the parameters.
         *
         * is_experience<M>::value will be equal to true is M implements the interface,
         * and false otherwise.
         *
         * @tparam M The class to test for the interface.
         */
        template <typename M>
        struct is_experience {
            private:
                template <typename Z> static constexpr auto test(int) -> decltype(

                        static_cast<long unsigned   (Z::*)(size_t,size_t,size_t) const>  (&Z::getVisits),
                        static_cast<long unsigned   (Z::*)(size_t,size_t) const>         (&Z::getVisitsSum),
                        static_cast<double          (Z::*)(size_t,size_t,size_t) const>  (&Z::getReward),
                        static_cast<double          (Z::*)(size_t,size_t) const>         (&Z::getRewardSum),

                        bool()
                ) { return true; }

                template <typename> static constexpr auto test(...) -> bool
                { return false; }

            public:
                enum { value = test<M>(0) };
        };
    }
}

#endif
