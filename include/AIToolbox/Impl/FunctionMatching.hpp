#ifndef AI_TOOLBOX_IMPL_FUNCTION_MATCHIN_HEADER_FILE
#define AI_TOOLBOX_IMPL_FUNCTION_MATCHIN_HEADER_FILE

#include <tuple>
#include <AIToolbox/Utils/TypeTraits.hpp>

namespace AIToolbox::Impl {
    /**
     * @name Function Matcher
     *
     * In this file we implement some metaprogramming goodies to improve the
     * interface of some things.
     *
     * In particular, here we implement a way to call a function which has a
     * subset of the arguments that we actually want to pass (provided they are
     * in the correct order). For example, say we want to call:
     *
     *     f(10, 0.5, 'c')
     *
     * But we are passed a function
     *
     *     void f(char);
     *
     * Then we would like to simply call
     *
     *    f('c');
     *
     * This is useful since it allows the user to implement callbacks with only
     * the parameters they care about, shortening interfaces a bit.
     *
     * @{
     */

    /**
     * @brief This struct helps decompose a function into return value and arguments.
     */
    template <typename T>
    struct GetFunctionArguments;

    template <typename R, typename... Args>
    struct GetFunctionArguments<R(*)(Args...)> {
        using args = std::tuple<Args...>;
        using rettype = R;
    };

    template <typename R, typename C, typename... Args>
    struct GetFunctionArguments<R(C::*)(Args...)> {
        using args = std::tuple<Args...>;
        using rettype = R;
    };

    template <typename R, typename C, typename... Args>
    struct GetFunctionArguments<R(C::*)(Args...) const> : GetFunctionArguments<R(C::*)(Args...)> {};

    /**
     * @brief This class is simply a template container for ids.
     */
    template <size_t... IDs> struct IdPack {};

    /**
     * @brief This struct allows to match between two tuples types.
     *
     * If the first tuple is an ordered subset of the second, match will be
     * true and the type will be an IdPack with the indeces of the second tuple
     * that match the first.
     *
     * Otherwise, match will be false and type will be void.
     *
     * You can use this by simply calling
     *
     *     Matcher<0, smaller_tuple_type, bigger_tuple_type>::match
     *     Matcher<0, smaller_tuple_type, bigger_tuple_type>::type
     */
    template <size_t N, typename T, typename U, size_t... IDs>
    struct Matcher {
        static constexpr bool match = false;
        using type = IdPack<IDs...>;
    };

    template <size_t N, typename... B, size_t... IDs>
    struct Matcher<N, std::tuple<>, std::tuple<B...>, IDs...> {
        static constexpr bool match = true;
        using type = IdPack<IDs...>;
    };

    template <size_t N, typename F, typename... A, typename... B, size_t... IDs>
    struct Matcher<N, std::tuple<F, A...>, std::tuple<F, B...>, IDs...> {
        using M = Matcher<N+1, std::tuple<A...>, std::tuple<B...>, IDs..., N>;
        static constexpr bool match = M::match;
        using type = typename M::type;
    };

    template <size_t N, typename FA, typename... A, typename FB, typename... B, size_t... IDs>
    struct Matcher<N, std::tuple<FA, A...>, std::tuple<FB, B...>, IDs...> {
        using M = std::conditional_t<
                    std::is_constructible_v<FA, FB> &&
                    std::is_same_v<remove_cv_ref_t<FA>, remove_cv_ref_t<FB>>,
                    Matcher<N+1, std::tuple<A...>, std::tuple<B...>, IDs..., N>,
                    Matcher<N+1, std::tuple<FA, A...>, std::tuple<B...>, IDs...>
                >;
        static constexpr bool match = M::match;
        using type = typename M::type;
    };

    /**
     * @brief This struct reports whether a given function is compatible with a given signature.
     *
     * We check that the return value is the same, and that the argument list
     * of the function type to check (first argument) is a subset of the longer
     * one (second argument).
     *
     * @tparam T The function type to check.
     * @tparam F The longer function type to compare against.
     */
    template <typename T, typename F>
    struct is_compatible_f {
        static constexpr bool value = false;
    };

    template <typename R, typename... Args, typename R2, typename... Args2>
    struct is_compatible_f<R(Args...), R2(Args2...)> {
        static constexpr bool value = std::is_same<R, R2>::value &&
                                      Matcher<0, std::tuple<Args...>, std::tuple<Args2...>>::match;
    };

    template <typename R, typename C, typename... Args, typename R2, typename... Args2>
    struct is_compatible_f<R(C::*)(Args...), R2(Args2...)> : is_compatible_f<R(Args...), R2(Args2...)> {};

    template <typename R, typename C, typename... Args, typename R2, typename... Args2>
    struct is_compatible_f<R(C::*)(Args...) const, R2(Args2...)> : is_compatible_f<R(C::*)(Args...), R2(Args2...)> {};

    /**
     * @brief This function calls the input function with the subset of correct parameters from the input tuple.
     *
     * @param f The function to call.
     * @param args All arguments.
     * @param IdPack The type containing the indeces of the arguments to pass to the function.
     */
    template <typename F, typename... Args, size_t... IDs>
    void caller(F f, std::tuple<Args...> && args, IdPack<IDs...>) {
        f(std::forward<std::tuple_element_t<IDs, std::tuple<Args...>>>(std::get<IDs>(args))...);
    }

    template <typename C, typename F, typename... Args, size_t... IDs>
    void caller(C & c, F f, std::tuple<Args...> && args, IdPack<IDs...>) {
        (c.*f)(std::forward<std::tuple_element_t<IDs, std::tuple<Args...>>>(std::get<IDs>(args))...);
    }

    /**
     * @brief This function calls the input function with the specified arguments.
     *
     * If the function only accepts a subset of the input arguments, only the
     * correct ones are passed through. We do require that the arguments are in
     * the same order (we don't find permutations of the arguments which might
     * match).
     *
     * @param f The function to call.
     * @param ...args The arguments to pass.
     */
    template <typename F, typename... Args>
    void callFunction(F f, Args&& ...args) {
        using FArgs = typename GetFunctionArguments<F>::args;
        using IdList = typename Matcher<0, FArgs, std::tuple<Args...>>::type;

        caller(f, std::forward_as_tuple(args...), IdList());
    }

    /**
     * @brief This function calls the input member function with the specified arguments.
     *
     * If the function only accepts a subset of the input arguments, only the
     * correct ones are passed through. We do require that the arguments are in
     * the same order (we don't find permutations of the arguments which might
     * match).
     *
     * @param c The object to call the member function on.
     * @param f The member function to call.
     * @param ...args The arguments to pass.
     */
    template <typename C, typename F, typename... Args>
    void callFunction(C & c, F f, Args&& ...args) {
        using FArgs = typename GetFunctionArguments<F>::args;
        using IdList = typename Matcher<0, FArgs, std::tuple<Args...>>::type;
        static_assert(Matcher<0, FArgs, std::tuple<Args...>>::match);

        caller(c, f, std::forward_as_tuple(args...), IdList());
    }

    /** @}  */
}

#endif
