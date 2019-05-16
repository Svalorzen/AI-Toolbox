#ifndef AI_TOOLBOX_MDP_TYPE_TRAITS_HEADER_FILE
#define AI_TOOLBOX_MDP_TYPE_TRAITS_HEADER_FILE

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
    template <typename M, typename ST, typename AT>
    struct is_generative_model {
        private:
            template <typename Z> static constexpr auto test(int) -> decltype(

                    static_cast<size_t (Z::*)() const>                                      (&Z::getS),
                    static_cast<size_t (Z::*)() const>                                      (&Z::getA),
                    static_cast<double (Z::*)() const>                                      (&Z::getDiscount),
                    static_cast<std::tuple<ST, double> (Z::*)(ST,AT) const>                 (&Z::sampleSR),
                    static_cast<bool (Z::*)(ST) const>                                      (&Z::isTerminal),
                    static_cast<std::vector<AT> (Z::*)(ST) const>                           (&Z::getAllowedActions),

                    bool()
            ) { return true; }

            template <typename Z> static constexpr auto test(...) -> bool
            { return false; }

        public:
            enum { value = test<M>(0) };
    };
    template <typename M, typename ST, typename AT>
    inline constexpr bool is_generative_model_v = is_generative_model<M, ST, AT>::value;

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
    template <typename M, typename ST = size_t, typename AT = size_t>
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
            enum { value = test<M>(0) && is_generative_model_v<M, ST, AT> };
    };
    template <typename M>
    inline constexpr bool is_model_v = is_model<M>::value;

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
    template <typename M>
    inline constexpr bool is_experience_v = is_experience<M>::value;
}

#endif
