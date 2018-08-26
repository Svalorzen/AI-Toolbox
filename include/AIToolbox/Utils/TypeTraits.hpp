#ifndef AI_TOOLBOX_TYPE_TRAITS_HEADER_FILE
#define AI_TOOLBOX_TYPE_TRAITS_HEADER_FILE

#include <type_traits>
#include <utility>

namespace AIToolbox {
    namespace Impl {
        template <typename Iterator, typename Check = void>
        struct IterSwap {
            void operator()(Iterator lhs, Iterator rhs) {
                using std::swap;
                swap(*lhs, *rhs);
            }
        };

        template <typename Iterator>
        struct IterSwap<Iterator, decltype(std::declval<Iterator*>()->base(), void())> {
            void operator()(Iterator lhs, Iterator rhs) {
                using std::swap;
                swap(*(lhs.base()), *(rhs.base()));
            }
        };
    }

    /**
     * @brief This function swaps the objects pointed by the two iterators.
     *
     * This function is needed in order to be able to treat in the same way
     * normal iterators and proxy iterators (such as
     * boost::transform_iterator). This allows us to write algorithms that
     * operate on a specific part of the data, but can alter the original range
     * as needed.
     */
    template <typename Iterator>
    void iter_swap(Iterator lhs, Iterator rhs) {
        return Impl::IterSwap<Iterator>()(lhs, rhs);
    }

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
}

#endif
