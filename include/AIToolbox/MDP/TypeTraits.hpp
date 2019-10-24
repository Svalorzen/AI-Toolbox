#ifndef AI_TOOLBOX_MDP_TYPE_TRAITS_HEADER_FILE
#define AI_TOOLBOX_MDP_TYPE_TRAITS_HEADER_FILE

#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox::MDP {
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
    template <typename M>
    inline constexpr bool is_generative_model_v = is_generative_model<M>::value;

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
            enum { value = test<M>(0) && is_generative_model_v<M> };
    };
    template <typename M>
    inline constexpr bool is_model_v = is_model<M>::value;

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
     * - R getRewardFunction() const : Returns the reward function as a matrix SxA', where R is some Eigen matrix type.
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
            // With this macro we can find out the return type of a given member function; we use it
            // so that we can check whether the class offers methods which return Eigen types, so we
            // can enable the high-performance algorithm variants.
            #define RETVAL_EXTRACTOR(fun_name)                                                                                      \
                                                                                                                                    \
            template <typename Z, typename ...Args> static auto fun_name##RetType(Z* z) ->                                          \
                                                                remove_cv_ref_t<decltype(z->fun_name(std::declval<Args>()...))>;    \
                                                                                                                                    \
            template <typename Z, typename ...Args> static auto fun_name##RetType(...) -> int

            RETVAL_EXTRACTOR(getTransitionFunction);
            RETVAL_EXTRACTOR(getRewardFunction);

            // The template parameters here must match the ones used in the test function!
            // So const M if the function is const, and then the parameter types.
            using F = decltype(getTransitionFunctionRetType<const M, size_t>(0));
            using R = decltype(getRewardFunctionRetType<const M>(0));

            template <typename Z> static constexpr auto test(int) -> decltype(

                    static_cast<const F & (Z::*)(size_t) const>         (&Z::getTransitionFunction),
                    static_cast<const R & (Z::*)()       const>         (&Z::getRewardFunction),

                    bool()
            ) { return true; }

            template <typename Z> static constexpr auto test(...) -> bool
            { return false; }

            #undef RETVAL_EXTRACTOR

        public:
            enum {
                value = is_model_v<M> && test<M>(0) &&
                        std::is_base_of_v<Eigen::EigenBase<F>, F> &&
                        std::is_base_of_v<Eigen::EigenBase<R>, R>
            };
    };
    template <typename M>
    inline constexpr bool is_model_eigen_v = is_model_eigen<M>::value;

    /**
     * @brief This struct verifies that a class satisfies the is_model interface but not the is_model_eigen interface.
     *
     * @tparam M The class to test for the interface.
     */
    template <typename M>
    struct is_model_not_eigen {
        public:
            enum { value = is_model_v<M> && !is_model_eigen_v<M> };
    };
    template <typename M>
    inline constexpr bool is_model_not_eigen_v = is_model_not_eigen<M>::value;

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
                    static_cast<double          (Z::*)(size_t,size_t) const>  (&Z::getReward),
                    static_cast<double          (Z::*)(size_t,size_t) const>  (&Z::getM2),

                    bool()
            ) { return true; }

            template <typename> static constexpr auto test(...) -> bool
            { return false; }

        public:
            enum { value = test<M>(0) };
    };
    template <typename M>
    inline constexpr bool is_experience_v = is_experience<M>::value;

    /**
     * @brief This struct represents the required Experience interface that allows leverage Eigen.
     *
     * This struct is used to check interfaces of classes in templates.  In
     * particular, this struct tests for the interface of an MDP Experience
     * which uses Eigen matrices internally.
     * The interface must be implemented and be public in the parameter
     * class. The interface is the following:
     *
     * - T getVisitsTable(size_t a) const : Returns the visits table for a given action as a matrix SxS', where T is some Eigen matrix type.
     * - R getRewardFunction() const : Returns the reward matrix as a matrix SxA', where R is some Eigen matrix type.
     * - S getM2Matrix() const : Returns the M2 matrix as a matrix SxA', where S is some Eigen matrix type.
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
    template <typename E>
    struct is_experience_eigen {
        private:
            // With this macro we can find out the return type of a given member function; we use it
            // so that we can check whether the class offers methods which return Eigen types, so we
            // can enable the high-performance algorithm variants.
            #define RETVAL_EXTRACTOR(fun_name)                                                                                      \
                                                                                                                                    \
            template <typename Z, typename ...Args> static auto fun_name##RetType(Z* z) ->                                          \
                                                                remove_cv_ref_t<decltype(z->fun_name(std::declval<Args>()...))>;    \
                                                                                                                                    \
            template <typename Z, typename ...Args> static auto fun_name##RetType(...) -> int

            RETVAL_EXTRACTOR(getVisitsTable);
            RETVAL_EXTRACTOR(getRewardMatrix);
            RETVAL_EXTRACTOR(getM2Matrix);

            // The template parameters here must match the ones used in the test function!
            // So const M if the function is const, and then the parameter types.
            using F = decltype(getVisitsTableRetType<const E, size_t>(0));
            using R = decltype(getRewardMatrixRetType<const E>(0));
            using S = decltype(getM2MatrixRetType<const E>(0));

            template <typename Z> static constexpr auto test(int) -> decltype(

                    static_cast<const F & (Z::*)(size_t) const>         (&Z::getVisitsTable),
                    static_cast<const R & (Z::*)()       const>         (&Z::getRewardMatrix),
                    static_cast<const S & (Z::*)()       const>         (&Z::getM2Matrix),

                    bool()
            ) { return true; }

            template <typename Z> static constexpr auto test(...) -> bool
            { return false; }

            #undef RETVAL_EXTRACTOR

        public:
            enum {
                value = is_experience_v<E> && test<E>(0) &&
                        std::is_base_of_v<Eigen::EigenBase<F>, F> &&
                        std::is_base_of_v<Eigen::EigenBase<R>, R> &&
                        std::is_base_of_v<Eigen::EigenBase<S>, S>
            };
    };
    template <typename M>
    inline constexpr bool is_experience_eigen_v = is_experience_eigen<M>::value;
}

#endif
