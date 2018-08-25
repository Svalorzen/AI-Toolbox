#ifndef AI_TOOLBOX_TYPE_TRAITS_HEADER_FILE
#define AI_TOOLBOX_TYPE_TRAITS_HEADER_FILE

#include <type_traits>
#include <utility>

namespace AIToolbox {
    namespace Impl {
        template <typename Iterator, typename Check = void>
        struct BaseIter {
            using BaseIterator = Iterator;
            Iterator operator()(Iterator it) { return it; }
        };

        template <typename Iterator>
        struct BaseIter<Iterator, decltype(std::declval<Iterator*>()->base(), void())> {
            using BaseIterator = decltype(std::declval<Iterator>().base());
            BaseIterator operator()(const Iterator & it) { return it.base(); }
        };
    }

    /**
     * @brief This function returns the base iterator for any given iterator.
     *
     * A base iterator exists if the iterator implements the method base(). If
     * not, a copy of the iterator itself is returned.
     *
     * @param it The iterator to return the base of.
     *
     * @return The base iterator of the input.
     */
    template <typename Iterator>
    typename Impl::BaseIter<typename std::remove_reference<Iterator>::type>::BaseIterator baseIter(Iterator && it) {
        return Impl::BaseIter<typename std::remove_reference<Iterator>::type>()(std::forward<Iterator>(it));
    }

    /**
     * @brief This struct is used to copy constness from one type to another.
     */
    template <typename CopiedType, typename ConstReference>
    struct copy_const {
        using type = typename std::conditional<std::is_const<ConstReference>::value,
                                      typename std::add_const<CopiedType>::type,
                                      typename std::remove_const<CopiedType>::type>::type;
    };
    template <typename CopiedType, typename ConstReference>
    using copy_const_t = typename copy_const<CopiedType, ConstReference>::type;

    /**
     * @brief This struct is used to both remove references and all cv qualifiers.
     */
    template <typename T>
    struct remove_cv_ref { using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type; };
    template <typename T>
    using remove_cv_ref_t = typename remove_cv_ref<T>::type;
}

#endif
