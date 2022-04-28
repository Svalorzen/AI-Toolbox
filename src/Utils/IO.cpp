#include <AIToolbox/Utils/IO.hpp>

#include <limits>
#include <iomanip>

#include <AIToolbox/Logging.hpp>

namespace AIToolbox {
    // ################################################
    // #################### OUTPUT ####################
    // ################################################

    std::ostream & write(std::ostream & os, double d) {
        const auto oldPrecision = os.precision(std::numeric_limits<double>::max_digits10);

        os << d << '\n';

        os.precision(oldPrecision);
        return os;
    }

    std::ostream & write(std::ostream & os, const Vector & v) {
        const auto oldPrecision = os.precision(std::numeric_limits<double>::max_digits10);

        for (size_t i = 0; i < static_cast<size_t>(v.size()); ++i)
            os << v[i] << ' ';
        os << '\n';

        os.precision(oldPrecision);
        return os;
    }

    std::ostream & write(std::ostream & os, const Matrix2D & m) {
        const auto oldPrecision = os.precision(std::numeric_limits<double>::max_digits10);

        for (size_t i = 0; i < static_cast<size_t>(m.rows()); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(m.cols()); ++j)
                os << m(i, j) << ' ';
            os << '\n';
        }
        os << '\n';

        os.precision(oldPrecision);
        return os;
    }

    std::ostream & write(std::ostream & os, const SparseMatrix2D & m) {
        const auto oldPrecision = os.precision(std::numeric_limits<double>::max_digits10);

        // We need to first compute how many non-zero entries we have.
        size_t entriesNum = 0;
        for (int k = 0; k < m.outerSize(); ++k)
            for (SparseMatrix2D::InnerIterator it(m, k); it; ++it)
                ++entriesNum;

        os << entriesNum << '\n';
        for (int k = 0; k < m.outerSize(); ++k)
            for (SparseMatrix2D::InnerIterator it(m, k); it; ++it)
                os << it.row() << ' ' << it.col() << ' ' << it.value() << '\n';

        os.precision(oldPrecision);
        return os;
    }

    std::ostream & write(std::ostream & os, const Matrix3D & m) {
        for (size_t i = 0; i < m.size(); ++i)
            write(os, m[i]);
        return os;
    }

    std::ostream & write(std::ostream & os, const SparseMatrix3D & m) {
        for (size_t i = 0; i < m.size(); ++i)
            write(os, m[i]);
        return os;
    }

    std::ostream & write(std::ostream & os, const Table2D & t) {
        for (size_t i = 0; i < static_cast<size_t>(t.rows()); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(t.cols()); ++j)
                os << t(i, j) << ' ';
            os << '\n';
        }
        os << '\n';
        return os;
    }

    std::ostream & write(std::ostream & os, const SparseTable2D & t) {
        // We need to first compute how many non-zero entries we have.
        size_t entriesNum = 0;
        for (int k = 0; k < t.outerSize(); ++k)
            for (SparseTable2D::InnerIterator it(t, k); it; ++it)
                ++entriesNum;

        os << entriesNum << '\n';
        for (int k = 0; k < t.outerSize(); ++k)
            for (SparseTable2D::InnerIterator it(t, k); it; ++it)
                os << it.row() << ' ' << it.col() << ' ' << it.value() << '\n';

        return os;
    }

    std::ostream & write(std::ostream & os, const Table3D & t) {
        for (size_t i = 0; i < t.size(); ++i)
            write(os, t[i]);
        return os;
    }

    std::ostream & write(std::ostream & os, const SparseTable3D & t) {
        for (size_t i = 0; i < t.size(); ++i)
            write(os, t[i]);
        return os;
    }

    // ###############################################
    // #################### INPUT ####################
    // ###############################################

    std::istream & read(std::istream & is, Vector & v) {
        Vector in(v.size());
        for (size_t i = 0; i < static_cast<size_t>(v.size()); ++i) {
            if ( !(is >> in[i]) ) {
                AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Vector data, element " << i << " out of " << v.size());
                break;
            }
        }
        if (is) v = std::move(in);
        return is;
    }

    std::istream & read(std::istream & is, Matrix2D & m) {
        Matrix2D in(m.rows(), m.cols());
        for (size_t i = 0; i < static_cast<size_t>(m.rows()); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(m.cols()); ++j) {
                if ( !(is >> in(i, j)) ) {
                    AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Matrix2D data, element (" << i << ',' << j << " out of " << m.rows() << ',' << m.cols());
                    break;
                }
            }
        }
        if (is) m = std::move(in);
        return is;
    }

    std::istream & read(std::istream & is, SparseMatrix2D & m) {
        std::vector<Eigen::Triplet<double>> in;
        size_t toRead;
        if ( !(is >> toRead) ) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read the number of non-zero entries for SparseMatrix2D");
            return is;
        }
        if (toRead > static_cast<size_t>(m.rows() * m.cols())) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Too many entries to read for SparseMatrix2D");
            is.setstate(std::ios::failbit);
            return is;
        }

        size_t r, c; double v;
        for (size_t i = 0; i < toRead; ++i) {
            if ( !(is >> r >> c >> v) ) {
                AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseMatrix2D data, element " << i << " out of " << toRead);
                break;
            }
            if (r >= static_cast<size_t>(m.rows())) {
                AI_LOGGER(AI_SEVERITY_ERROR, "Invalid row index while reading SparseMatrix2D data, element " << i << " out of " << toRead);
                is.setstate(std::ios::failbit);
                break;
            }
            if (c >= static_cast<size_t>(m.cols())) {
                AI_LOGGER(AI_SEVERITY_ERROR, "Invalid column index while reading SparseMatrix2D data, element " << i << " out of " << toRead);
                is.setstate(std::ios::failbit);
                break;
            }
            in.emplace_back(r, c, v);
        }
        if (is) m.setFromTriplets(std::begin(in), std::end(in));
        return is;
    }

    std::istream & read(std::istream & is, Matrix3D & m) {
        Matrix3D in; in.reserve(m.size());
        for (size_t i = 0; i < m.size(); ++i) {
            Matrix2D inHelper(m[i].rows(), m[i].cols());
            if (!read(is, inHelper)) {
                AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Matrix3D data, matrix " << i << " out of " << m.size());
                break;
            }
            in.push_back(std::move(inHelper));
        }
        if (is) m = std::move(in);
        return is;
    }

    std::istream & read(std::istream & is, SparseMatrix3D & m) {
        SparseMatrix3D in; in.reserve(m.size());
        for (size_t i = 0; i < m.size(); ++i) {
            SparseMatrix2D inHelper(m[i].rows(), m[i].cols());
            if (!read(is, inHelper)) {
                AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseMatrix3D data, matrix " << i << " out of " << m.size());
                break;
            }
            in.push_back(std::move(inHelper));
        }
        if (is) m = std::move(in);
        return is;
    }

    std::istream & read(std::istream & is, Table2D & t) {
        Table2D in(t.rows(), t.cols());
        for (size_t i = 0; i < static_cast<size_t>(t.rows()); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(t.cols()); ++j) {
                if ( !(is >> in(i, j)) ) {
                    AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Table2D data, element (" << i << ',' << j << " out of " << t.rows() << ',' << t.cols());
                    break;
                }
            }
        }
        if (is) t = std::move(in);
        return is;
    }

    std::istream & read(std::istream & is, SparseTable2D & t) {
        std::vector<Eigen::Triplet<double>> in;
        size_t toRead;
        if ( !(is >> toRead) ) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read the number of non-zero entries for SparseTable2D");
            return is;
        }
        if (toRead > static_cast<size_t>(t.rows() * t.cols())) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Too many entries to read for SparseTable2D");
            is.setstate(std::ios::failbit);
            return is;
        }

        size_t r, c; double v;
        for (size_t i = 0; i < toRead; ++i) {
            if ( !(is >> r >> c >> v) ) {
                AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseTable2D data, element " << i << " out of " << toRead);
                break;
            }
            if (r >= static_cast<size_t>(t.rows())) {
                AI_LOGGER(AI_SEVERITY_ERROR, "Invalid row index while reading SparseTable2D data, element " << i << " out of " << toRead);
                is.setstate(std::ios::failbit);
                break;
            }
            if (c >= static_cast<size_t>(t.cols())) {
                AI_LOGGER(AI_SEVERITY_ERROR, "Invalid column index while reading SparseTable2D data, element " << i << " out of " << toRead);
                is.setstate(std::ios::failbit);
                break;
            }
            in.emplace_back(r, c, v);
        }
        if (is) t.setFromTriplets(std::begin(in), std::end(in));
        return is;
    }

    std::istream & read(std::istream & is, Table3D & t) {
        Table3D in; in.reserve(t.size());
        for (size_t i = 0; i < t.size(); ++i) {
            Table2D inHelper(t[i].rows(), t[i].cols());
            if (!read(is, inHelper)) {
                AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Table3D data, table " << i << " out of " << t.size());
                break;
            }
            in.push_back(std::move(inHelper));
        }
        if (is) t = std::move(in);
        return is;
    }

    std::istream & read(std::istream & is, SparseTable3D & t) {
        SparseTable3D in; in.reserve(t.size());
        for (size_t i = 0; i < t.size(); ++i) {
            SparseTable2D inHelper(t[i].rows(), t[i].cols());
            if (!read(is, inHelper)) {
                AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseTable3D data, table " << i << " out of " << t.size());
                break;
            }
            in.push_back(std::move(inHelper));
        }
        if (is) t = std::move(in);
        return is;
    }
}
