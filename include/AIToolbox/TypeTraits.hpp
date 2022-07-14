#ifndef AI_TOOLBOX_TYPE_TRAITS_HEADER_FILE
#define AI_TOOLBOX_TYPE_TRAITS_HEADER_FILE

#include <type_traits>
#include <utility>
#include <concepts>
#include <cstddef>

#include <Eigen/Core>

namespace AIToolbox {
    /**
     * @brief This struct is used to copy constness from one type to another.
     */
    template <typename CopiedType, typename ConstReference>
    struct copy_const {
        using type = typename std::conditional_t<std::is_const_v<ConstReference>,
                                      std::add_const_t<CopiedType>,
                                      std::remove_const_t<CopiedType>>;
    };
    template <typename CopiedType, typename ConstReference>
    using copy_const_t = typename copy_const<CopiedType, ConstReference>::type;

    /**
     * @brief This concept simplifies checking for non-void.
     */
    template <typename A, typename B>
    concept different_from = !std::same_as<A, B>;

    /**
     * @brief This concept checks for a simple 2D accessible matrix.
     */
    template <typename M>
    concept IsNaive2DMatrix = requires (M m) {
        { m[0][0] } -> std::convertible_to<double>;
    };

    /**
     * @brief This concept checks for a simple 3D accessible matrix.
     */
    template <typename M>
    concept IsNaive3DMatrix = requires (M m) {
        { m[0][0][0] } -> std::convertible_to<double>;
    };

    /**
     * @brief This concept checks for a simple 2D accessible table.
     */
    template <typename T>
    concept IsNaive2DTable = requires (T t) {
        { t[0][0] } -> std::convertible_to<long unsigned>;
    };

    /**
     * @brief This concept checks for a simple 3D accessible table.
     */
    template <typename T>
    concept IsNaive3DTable = requires (T t) {
        { t[0][0][0] } -> std::convertible_to<long unsigned>;
    };

    /**
     * @brief This concept simplifies checking for non-void.
     */
    template <typename T>
    concept IsDerivedFromEigen = std::derived_from<T, Eigen::EigenBase<T>>;

    // #############################################
    // ######## Generic modeling concepts ##########
    // #############################################

    /**
     * @brief This concept checks that getS() exists.
     */
    template <typename M>
    concept HasStateSpace = requires(const M m) {
        { m.getS() } -> different_from<void>;
    };

    /**
     * @brief This concept checks that getA() exists.
     */
    template <typename M>
    concept HasFixedActionSpace = requires(const M m) {
        { m.getA() } -> different_from<void>;
    };

    /**
     * @brief This concept checks that getA(state) exists.
     */
    template <typename M>
    concept HasVariableActionSpace = requires(const M m) {
        { m.getA(m.getS()) } -> different_from<void>;
    };

    /**
     * @brief This concept checks that some form of getA exists.
     */
    template <typename M>
    concept HasActionSpace = HasFixedActionSpace<M> || HasVariableActionSpace<M>;

    /**
     * @brief This concept checks that getO() exists.
     */
    template <typename M>
    concept HasObservationSpace = requires(const M m) {
        { m.getO() } -> different_from<void>;
    };

    /**
     * @brief This concept checks that getS() returns size_t.
     */
    template <typename M>
    concept HasIntegralStateSpace = requires(const M m) {
        { m.getS() } -> std::same_as<size_t>;
    };

    /**
     * @brief This concept checks that getA returns size_t.
     */
    template <typename M>
    concept HasIntegralActionSpace = requires(const M m) {
        requires
          (HasFixedActionSpace<M>    && requires {{ m.getA() }         -> std::same_as<size_t>; }) ||
          (HasVariableActionSpace<M> && requires {{ m.getA(m.getS()) } -> std::same_as<size_t>; });
    };

    /**
     * @brief This concept checks that getO() returns size_t.
     */
    template <typename M>
    concept HasIntegralObservationSpace = requires(const M m) {
        { m.getO() } -> std::same_as<size_t>;
    };

    /**
     * @brief This concept checks the minimum requirements for a generative model.
     *
     * This concept defines the minimum requirements for the minimum generative
     * model we will probably accept around the library.
     *
     * While the concept has no specific requirements for what the states and
     * actions can be, each algorithm will probably be more strict about types
     * (for example may require integral state/action spaces, or a fixed action
     * space).
     *
     * We need:
     *
     * - getS()
     * - getA() or getA(state)
     *
     * Note that we do not specify here which types the states and actions should be.
     *
     * - getDiscount()
     * - sampleSR(state, action)
     * - isTerminal(state)
     */
    template <typename M>
    concept IsGenerativeModel = HasStateSpace<M> && HasActionSpace<M> && requires(const M m) {
        { m.getDiscount() }                -> std::convertible_to<double>;
        { m.isTerminal(m.getS()) }         -> std::convertible_to<bool>;

        requires
          (HasFixedActionSpace<M>    && requires {{ m.sampleSR(m.getS(), m.getA()) }         -> std::convertible_to<std::tuple<std::remove_cvref_t<decltype(m.getS())>, double>>; }) ||
          (HasVariableActionSpace<M> && requires {{ m.sampleSR(m.getS(), m.getA(m.getS())) } -> std::convertible_to<std::tuple<std::remove_cvref_t<decltype(m.getS())>, double>>; });
    };
}

#endif
