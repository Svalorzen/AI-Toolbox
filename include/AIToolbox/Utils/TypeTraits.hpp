#ifndef AI_TOOLBOX_TYPE_TRAITS_HEADER_FILE
#define AI_TOOLBOX_TYPE_TRAITS_HEADER_FILE

#include <type_traits>
#include <utility>

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
     * @brief This struct is used to both remove references and all cv qualifiers.
     */
    template <typename T>
    struct remove_cv_ref { using type = std::remove_cv_t<std::remove_reference_t<T>>; };
    template <typename T>
    using remove_cv_ref_t = typename remove_cv_ref<T>::type;

    /**
     * @brief Equivalent of C++20 std::identity.
     *
     * We use this as a standard projection.
     */
    struct identity {
        template< class T>
        constexpr T&& operator()( T&& t ) const noexcept {
            return std::forward<T>(t);
        }
        using is_transparent = void;
    };
}

#endif
