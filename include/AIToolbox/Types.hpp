#ifndef AI_TOOLBOX_TYPES_HEADER_FILE
#define AI_TOOLBOX_TYPES_HEADER_FILE

#include <vector>
#include <unordered_map>
#include <random>

#include <boost/multi_array.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace AIToolbox {
    // This should have decent properties.
    using RandomEngine = std::mt19937;

    using Table3D = boost::multi_array<double, 3>;
    using Table2D = boost::multi_array<double, 2>;

    using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    using Matrix2D           = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;
    using SparseMatrix2D     = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using SparseMatrix2DLong = Eigen::SparseMatrix<long,   Eigen::RowMajor>;

    using Matrix3D           = std::vector<Matrix2D>;
    using SparseMatrix3D     = std::vector<SparseMatrix2D>;
    using SparseMatrix3DLong = std::vector<SparseMatrix2DLong>;

    using Matrix4D       = boost::multi_array<Matrix2D,       2>;
    using SparseMatrix4D = boost::multi_array<SparseMatrix2D, 2>;

    // This is used to store a probability vector (sums to one, every element >= 0, <= 1)
    using ProbabilityVector = Vector;

    /**
     * @brief This is used to tag functions that avoid runtime checks.
     */
    inline struct NoCheck {} NO_CHECK;
}

#endif
