#ifndef AI_TOOLBOX_POMDP_TYPE_TRAITS_HEADER_FILE
#define AI_TOOLBOX_POMDP_TYPE_TRAITS_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This struct represents the required interface for a generative MDP.
     *
     * This struct is used to check interfaces of classes in templates.  In
     * particular, this struct tests for the interface of a generative MDP
     * model.  The interface must be implemented and be public in the
     * parameter class. The interface is the following:
     *
     * - std::tuple<size_t, size_t, double> sampleSOR(size_t s, size_t a) const : Returns a sampled state-observation-reward tuple from (s,a)
     *
     * is_generative_model<M>::value will be equal to true is M implements the interface,
     * and false otherwise.
     *
     * Note that, at least for now, we can avoid asking this generative
     * model for the total number of observation possible, because they are
     * not required as parameters for the functions, but just returned.
     * This may change in future though, depending on algorithms'
     * requirements.
     *
     * @tparam M The class to test for the interface.
     */
    template <typename M>
    struct is_generative_model {
        private:
            template <typename Z> static constexpr auto test(int) -> decltype(

                    static_cast<std::tuple<size_t,size_t,double> (Z::*)(size_t,size_t) const>      (&Z::sampleSOR),

                    bool()
            ) { return true; }

            template <typename Z> static constexpr auto test(...) -> bool
            { return false; }

        public:
            enum { value = test<M>(0) && MDP::is_generative_model_v<M> };
    };
    template <typename M>
    inline constexpr bool is_generative_model_v = is_generative_model<M>::value;

    /**
     * @brief This struct represents the required interface for a POMDP Model.
     *
     * This struct is used to check interfaces of classes in templates.
     * In particular, this struct tests for the interface of a POMDP model.
     * The interface must be implemented and be public in the parameter
     * class. The interface is the following:
     *
     * - size_t getO() const : Returns the number of observations of the Model.
     * - double getObservationProbability(size_t s1, size_t a, size_t o) const : Returns the probability for observation o after action a and final state s1.
     *
     * In addition the POMDP needs to respect the interface for the POMDP generative
     * model and the MDP model.
     *
     * \sa is_generative_model
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
            template <typename Z> static constexpr auto test(int) -> decltype(

                    static_cast<size_t (Z::*)() const>                      (&Z::getO),
                    static_cast<double (Z::*)(size_t,size_t,size_t) const>  (&Z::getObservationProbability),

                    bool()
            ) { return true; }

            template <typename> static constexpr auto test(...) -> bool
            { return false; }

        public:
            enum { value = test<T>(0) && is_generative_model_v<T> && MDP::is_model_v<T> };
    };
    template <typename M>
    inline constexpr bool is_model_v = is_model<M>::value;

    /**
     * @brief This struct represents the required interface that allows POMDP algorithms to leverage Eigen.
     *
     * This struct is used to check interfaces of classes in templates.
     * In particular, this struct tests for the interface of a POMDP model
     * which uses Eigen matrices internally.
     * The interface must be implemented and be public in the parameter
     * class. The interface is the following:
     *
     * - O getObservationFunction(size_t a) const : Returns the observation function for a given action as a matrix S'xO, where O is some Eigen matrix type.
     *
     * In addition the POMDP needs to respect the interface for the POMDP model
     * and the Eigen MDP model.
     *
     * \sa POMDP::is_model
     * \sa MDP::is_model_eigen
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

            RETVAL_EXTRACTOR(getObservationFunction);

            // The template parameters here must match the ones used in the test function!
            // So const M if the function is const, and then the parameter types.
            using O = decltype(getObservationFunctionRetType<const M, size_t>(0));

            template <typename Z> static constexpr auto test(int) -> decltype(

                    static_cast<const O & (Z::*)(size_t) const>         (&Z::getObservationFunction),

                    bool()
            ) { return true; }

            template <typename Z> static constexpr auto test(...) -> bool
            { return false; }

            #undef RETVAL_EXTRACTOR

        public:
            enum { value = is_model_v<M> && MDP::is_model_eigen_v<M> && test<M>(0) &&
                           std::is_base_of_v<Eigen::EigenBase<O>, O> };
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
}

#endif
