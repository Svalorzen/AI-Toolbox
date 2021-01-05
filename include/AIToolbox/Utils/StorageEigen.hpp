#ifndef AI_TOOLBOX_UTILS_STORAGE_EIGEN_HEADER_FILE
#define AI_TOOLBOX_UTILS_STORAGE_EIGEN_HEADER_FILE

#include <AIToolbox/Types.hpp>

namespace AIToolbox {
    /**
     * @brief This class provides an Eigen-compatible automatically resized Vector.
     *
     * The interface is deliberately simple for now. This class simply
     * maintains a public reference to the internal storage container.
     *
     * Pushing and popping modifies the view, while the storage gets increased
     * automatically when needed.
     */
    class StorageVector {
        private:
            Vector storage_;

        public:
            /**
             * @brief This member provides a view of the pushed data.
             */
            Eigen::Ref<Vector> vector;

            /**
             * @brief Basic constructor.
             *
             * @param startSize The initial pre-reserved space for storage.
             */
            StorageVector(size_t startSize);

            /**
             * @brief Basic constructor.
             *
             * @param vector The vector with which to initialize storage (and the public view).
             */
            StorageVector(Vector vector);

            /**
             * @brief This function removes elements from the view vector.
             *
             * @param num The number of elements to remove.
             */
            void pop_back(size_t num = 1);

            /**
             * @brief This function inserts a value in storage, and expands the view accordingly.
             *
             * @param val The value to push.
             */
            void push_back(double val);

            /**
             * @brief This function resizes the view to the requested size.
             *
             * We first call reserve(size_t) to make sure that the storage is
             * appropriately sized.
             *
             * @param size The requested new size.
             */
            void resize(size_t size);

            /**
             * @brief This function reserves space in the underlying storage.
             *
             * This function does not modify the view. We also call Eigen's
             * `conservativeResize` so that already stored data is maintained.
             *
             * @param size The requested new size.
             */
            void reserve(size_t size);
    };

    /**
     * @brief This class provides an Eigen-compatible automatically resized Matrix2D.
     *
     * The number of columns cannot be modified, only the number of rows.
     *
     * The interface is deliberately simple for now. This class simply
     * maintains a public reference to the internal storage container.
     *
     * Pushing and popping modifies the view, while the storage gets increased
     * automatically when needed.
     */
    class StorageMatrix2D {
        private:
            Matrix2D storage_;

        public:
            /**
             * @brief This member provides a view of the pushed data.
             */
            Eigen::Ref<Matrix2D> matrix;

            /**
             * @brief Basic constructor.
             *
             * @param startRows The initial pre-reserved rows for storage.
             * @param cols The fixed number of columns to store.
             */
            StorageMatrix2D(size_t startRows, size_t cols);

            /**
             * @brief Basic constructor.
             *
             * @param matrix The vector with which to initialize storage (and the public view).
             */
            StorageMatrix2D(Matrix2D matrix);

            /**
             * @brief This function removes rows from the view matrix.
             *
             * @param num The number of rows to remove.
             */
            void pop_back(size_t num = 1);

            /**
             * @brief This function inserts a new row in storage, and expands the view accordingly.
             *
             * The new row is left un-initialized. This function is provided
             * for performance, if the new row must be constructed dynamically.
             */
            void push_back();

            /**
             * @brief This function inserts a new row in storage, and expands the view accordingly.
             *
             * @param row The row to copy.
             */
            template <typename V>
            void push_back(const Eigen::MatrixBase<V> & row);

            /**
             * @brief This function resizes the view to the requested rows.
             *
             * We first call reserve(size_t) to make sure that the storage is
             * appropriately sized.
             *
             * @param size The requested new rows.
             */
            void resize(size_t rows);

            /**
             * @brief This function reserves row space in the underlying storage.
             *
             * This function does not modify the view. We also call Eigen's
             * `conservativeResize` so that already stored data is maintained.
             *
             * @param size The requested new rows.
             */
            void reserve(size_t rows);
    };

    template <typename V>
    void StorageMatrix2D::push_back(const Eigen::MatrixBase<V> & row) {
        const auto currRows = matrix.rows();

        if (storage_.rows() == currRows)
            storage_.conservativeResize(currRows * 2, Eigen::NoChange);

        storage_.row(currRows) = row;

        new (&matrix) Eigen::Ref<Matrix2D>(storage_.topRows(currRows + 1));
    }
}

#endif
