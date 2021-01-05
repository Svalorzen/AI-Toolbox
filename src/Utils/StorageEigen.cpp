#include <AIToolbox/Utils/StorageEigen.hpp>

namespace AIToolbox {
    // StorageVector

    StorageVector::StorageVector(const size_t startSize) :
            storage_(startSize), vector(storage_.head(0)) {}

    StorageVector::StorageVector(Vector vec) :
            storage_(std::move(vec)), vector(storage_) {}

    void StorageVector::pop_back(const size_t num) {
        const auto currSize = vector.size();
        assert(static_cast<size_t>(currSize) >= num);

        new (&vector) Eigen::Ref<Vector>(storage_.head(currSize - num));
    }

    void StorageVector::push_back(const double val) {
        const auto currSize = vector.size();

        if (storage_.size() == currSize)
            storage_.conservativeResize(currSize * 2);

        storage_[currSize] = val;

        new (&vector) Eigen::Ref<Vector>(storage_.head(currSize + 1));
    }

    void StorageVector::resize(const size_t size) {
        reserve(size);

        new (&vector) Eigen::Ref<Vector>(storage_.head(size));
    }

    void StorageVector::reserve(const size_t size) {
        if (storage_.size() < (long int)size)
            storage_.conservativeResize(size);
    }

    // StorageMatrix2D

    StorageMatrix2D::StorageMatrix2D(const size_t startRows, const size_t cols) :
            storage_(startRows, cols), matrix(storage_.topRows(0)) {}

    StorageMatrix2D::StorageMatrix2D(Matrix2D m) :
            storage_(std::move(m)), matrix(storage_) {}

    void StorageMatrix2D::pop_back(const size_t num) {
        const auto currRows = matrix.rows();
        assert(static_cast<size_t>(currRows) >= num);

        new (&matrix) Eigen::Ref<Matrix2D>(storage_.topRows(currRows - num));
    }

    void StorageMatrix2D::push_back() {
        const auto currRows = matrix.rows();

        if (storage_.rows() == currRows)
            storage_.conservativeResize(currRows * 2, Eigen::NoChange);

        new (&matrix) Eigen::Ref<Matrix2D>(storage_.topRows(currRows + 1));
    }

    void StorageMatrix2D::resize(const size_t rows) {
        reserve(rows);

        new (&matrix) Eigen::Ref<Matrix2D>(storage_.topRows(rows));
    }

    void StorageMatrix2D::reserve(const size_t rows) {
        if (storage_.rows() < (long int)rows)
            storage_.conservativeResize(rows, Eigen::NoChange);
    }
}
