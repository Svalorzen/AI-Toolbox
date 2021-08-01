#ifndef AI_TOOLBOX_UTILS_IO_HEADER_FILE
#define AI_TOOLBOX_UTILS_IO_HEADER_FILE

#include <AIToolbox/Types.hpp>

#include <iostream>

namespace AIToolbox {
    /**
     * @name Output stream utilities.
     *
     * These utilities output common types at the maximum possible precision.
     * This allows reading them back with no loss of accuracy.
     *
     * We do *not* output the size of the matrices, only their contents.
     *
     * We do not use operator<< here as the Eigen types already have it in
     * their own namespace, and we don't want ambiguities.
     *
     * @{
     */
    std::ostream & write(std::ostream & os, double d);

    std::ostream & write(std::ostream & os, const Vector & v);

    std::ostream & write(std::ostream & os, const Matrix2D & m);
    std::ostream & write(std::ostream & os, const SparseMatrix2D & m);

    std::ostream & write(std::ostream & os, const Matrix3D & m);
    std::ostream & write(std::ostream & os, const SparseMatrix3D & m);

    std::ostream & write(std::ostream & os, const Table2D & t);
    std::ostream & write(std::ostream & os, const SparseTable2D & t);

    std::ostream & write(std::ostream & os, const Table3D & t);
    std::ostream & write(std::ostream & os, const SparseTable3D & t);

    /** @}  */

    /**
     * @name Input stream utilities.
     *
     * These utilities read back data outputted with their respective
     * write() function.
     *
     * Note that the inputs must already be pre-allocated with the
     * correct size, as write() does not save this information.
     *
     * These functions do not modify the input if the parsing fails.
     *
     * @{
     */

    std::istream & read(std::istream & is, Vector & v);

    std::istream & read(std::istream & is, Matrix2D & m);
    std::istream & read(std::istream & is, SparseMatrix2D & m);

    std::istream & read(std::istream & is, Matrix3D & m);
    std::istream & read(std::istream & is, SparseMatrix3D & m);

    std::istream & read(std::istream & is, Table2D & t);
    std::istream & read(std::istream & is, SparseTable2D & t);

    std::istream & read(std::istream & is, Table3D & t);
    std::istream & read(std::istream & is, SparseTable3D & t);

    /** @}  */
}

#endif
