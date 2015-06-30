#ifndef AI_TOOLBOX_TYPES_HEADER_FILE
#define AI_TOOLBOX_TYPES_HEADER_FILE

#include <vector>
#include <unordered_map>

#include <boost/multi_array.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace AIToolbox {
    using Table3D = boost::multi_array<double, 3>;
    using Table2D = boost::multi_array<double, 2>;

    using Matrix2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;
    using SparseMatrix2D = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    using Vector   = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    using Matrix3D = std::vector<Matrix2D>;
    using SparseMatrix3D = std::vector<SparseMatrix2D>;
}

#endif
