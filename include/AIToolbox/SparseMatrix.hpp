#ifndef AI_TOOLBOX_SPARSE_MATRIX_HEADER_FILE
#define AI_TOOLBOX_SPARSE_MATRIX_HEADER_FILE

#include <tuple>
#include <vector>
#include <unordered_map>
#include <boost/functional/hash.hpp>

#include <AIToolbox/Utils.hpp>

namespace AIToolbox {
    template <size_t N, typename T, typename... Args>
    struct DimTupleImpl {
        using type = typename DimTupleImpl<N-1, T, Args..., T>::type;
    };

    template <typename T, typename... Args>
    struct DimTupleImpl<0, T, Args...> {
        using type = std::tuple<Args...>;
    };

    /**
     * @brief This struct's type field represents a tuple with N elements of the same type.
     *
     * @tparam N The number of elements of the tuple.
     * @tparam T The type of the elements of the tuple.
     */
    template <size_t N, typename T>
    struct DimTuple {
        using type = typename DimTupleImpl<N, T>::type;
    };

    template <typename T, typename U, typename... Args>
    struct are_all_convertible {
        static constexpr bool value = std::is_convertible<U, T>::value && are_all_convertible<T, Args...>::value;
    };

    template <typename T, typename U>
    struct are_all_convertible<T, U> {
        static constexpr bool value = std::is_convertible<U, T>::value;
    };

    /**
     * @brief This class represents a sparse matrix.
     *
     * Note that this is not supposed to be super-fancy, but unfortunately I could not
     * find any library which offered support for 3D sparse matrices. It contains
     * data in an unordered_map, which uses as keys the coordinates of the matrix as
     * tuples and doubles as values.
     *
     * It does not need to know the size of the matrix, since it really does not care about
     * them - it will not perform any bound checking. Instead, it just requires the number
     * of dimensions of the matrix, and if a value is requested outside the scope of the
     * matrix it will return 0.0, same for any element not stored in the class.
     *
     * @tparam N The dimensions of the matrix.
     */
    template <size_t N>
    class SparseMatrix {
        public:
            /**
             * @brief This function sets a value in the matrix.
             *
             * If the value is close enough to zero, it is simply removed from the matrix.
             * Otherwise it is added.
             *
             * @param v The value to assign to the matrix.
             * @param params The coordinate of the cell to assign, using integers (size_t).
             */
            template <typename... Params>
            void set(double v, Params... params) {
                static_assert(sizeof...(params) == N, "The supplied coordinate has the wrong number of dimensions");
                static_assert(are_all_convertible<size_t, Params...>::value, "Arguments are not convertible to size_t");

                if ( checkEqualSmall(v, 0.0) )
                    data_.erase(std::make_tuple((size_t)params...));
                else
                    data_[std::make_tuple((size_t)params...)] = v;
            }

            /**
             * @brief This function retrieves a value from the matrix.
             *
             * If the matrix does not contain the value it will return 0.0. Note that since the matrix
             * does not know of any supposed bounds, it will happily accept ANY coordinate (as long as
             * the dimensions are correct), and simply return zero for any values it does not contain.
             *
             * @param params The coordinate of the cell to assign, using integers (size_t).
             *
             * @return The value of the cell if stored by the class, 0.0 otherwise.
             */
            template <typename... Params>
            double operator()(Params... params) const {
                static_assert(sizeof...(params) == N, "The supplied coordinate has the wrong number of dimensions");
                static_assert(are_all_convertible<size_t, Params...>::value, "Arguments are not convertible to size_t");

                auto it = data_.find(std::make_tuple((size_t)params...));
                return it == data_.end() ? 0.0 : it->second;
            }

            /**
             * @brief This function retrieves a "row" of the matrix.
             *
             * Since this class has no concept of size for the matrix, only the dimensions, this function
             * takes as parameter the size of the row that is required, and the first N-1 coordinates.
             * It then builds a row containing the values stored in this matrix from 0 to size-1.
             * Elements not contained will be filled with zeroes.
             *
             * @param size The size of the row that is to be returned.
             * @param params The N-1 coordinates used to identify the row.
             *
             * @return A vector containing the values in the specified row of the matrix.
             */
            template <typename... Params>
            std::vector<double> getRow(size_t size, Params... params) const {
                static_assert(sizeof...(params) == N - 1, "The supplied coordinate has the wrong number of dimensions");
                static_assert(are_all_convertible<size_t, Params...>::value, "Arguments are not convertible to size_t");

                std::vector<double> v;
                v.reserve(size);

                for (size_t i = 0; i < size; ++i)
                    v.push_back(operator()(params..., i));

                return v;
            }

        private:
            std::unordered_map<typename DimTuple<N, size_t>::type, double, boost::hash<typename DimTuple<N, size_t>::type>> data_;
    };
}

#endif
