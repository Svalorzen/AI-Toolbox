#include "Utils.hpp"

#include <AIToolbox/Types.hpp>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

double getVectorItem(const AIToolbox::Vector& v, int index) {
    return v(index);
}

void setVectorItem(AIToolbox::Vector& v, int index, double value) {
    v(index) = value;
}

int getVectorLen(const AIToolbox::Vector& v) {
    return v.size();
}

double getMatrix2DItem(const AIToolbox::Matrix2D& m, boost::python::tuple i) {
    return m(boost::python::extract<int>(i[0]), boost::python::extract<int>(i[1]));
}

void setMatrix2DItem(AIToolbox::Matrix2D & m, boost::python::tuple i, double value) {
    m(boost::python::extract<int>(i[0]), boost::python::extract<int>(i[1])) = value;
}

boost::python::tuple getMatrix2DShape(const AIToolbox::Matrix2D& m) {
    return boost::python::make_tuple(m.rows(), m.cols());
}

struct VectorPickle : boost::python::pickle_suite {
    static boost::python::tuple getinitargs(const AIToolbox::Vector& v) {
        using namespace boost::python;
        return make_tuple(v.size());
    }

    static boost::python::tuple getstate(const AIToolbox::Vector& v) {
        using namespace boost::python;
        boost::python::list l;
        for (size_t i = 0; i < v.size(); ++i)
            l.append(v[i]);
        return make_tuple(l);
    }

    static void setstate(AIToolbox::Vector& v, boost::python::tuple state) {
        using namespace boost::python;
        if (len(state) != 1) {
            PyErr_SetObject(PyExc_ValueError,
                ("expected 1-item tuple in call to __setstate__; got %s" % state).ptr()
            );
            throw_error_already_set();
        }
        if (v.size() != len(state[0])) {
            PyErr_SetObject(PyExc_ValueError,
                ("state obtained in __setstate__ cannot be applied to this object; got %s" % state).ptr()
            );
            throw_error_already_set();
        }

        for (size_t i = 0; i < v.size(); ++i)
            v[i] = extract<double>(state[0][i]);
    }
};

struct Matrix2DPickle : boost::python::pickle_suite {
    static boost::python::tuple getinitargs(const AIToolbox::Matrix2D& m) {
        using namespace boost::python;
        return make_tuple(m.rows(), m.cols());
    }

    static boost::python::tuple getstate(const AIToolbox::Matrix2D& m) {
        using namespace boost::python;
        boost::python::list rows;
        for (size_t i = 0; i < m.rows(); ++i) {
            boost::python::list row;
            for (size_t j = 0; j < m.cols(); ++j)
                row.append(m(i, j));
            rows.append(row);
        }
        return make_tuple(rows);
    }

    static void setstate(AIToolbox::Matrix2D& m, boost::python::tuple state) {
        using namespace boost::python;
        if (len(state) != 1) {
            PyErr_SetObject(PyExc_ValueError,
                ("expected 1-item tuple in call to __setstate__; got %s" % state).ptr()
            );
            throw_error_already_set();
        }
        if (m.rows() != len(state[0])) {
            PyErr_SetObject(PyExc_ValueError,
                ("state obtained in __setstate__ cannot be applied to this object; got %s" % state).ptr()
            );
            throw_error_already_set();
        }

        for (size_t i = 0; i < m.rows(); ++i) {
            if (m.cols() != len(state[0][i])) {
                PyErr_SetObject(PyExc_ValueError,
                    ("state obtained in __setstate__ cannot be applied to this object; got %s" % state).ptr()
                );
                throw_error_already_set();
            }
            for (size_t j = 0; j < m.cols(); ++j)
                m(i, j) = extract<double>(state[0][i][j]);
        }
    }
};

void exportTypes() {
    using namespace AIToolbox;
    using namespace boost::python;

    // Eigen Vector
    class_<Vector>{"Vector", init<int>()}
        .def("__getitem__", &getVectorItem)
        .def("__setitem__", &setVectorItem)
        .def("__len__",     &getVectorLen)
        .def_pickle(VectorPickle());

    EigenVectorFromPython();

    // 2D Eigen matrix
    class_<Matrix2D>{"Matrix2D", init<int, int>()}
        .def("__getitem__", &getMatrix2DItem)
        .def("__setitem__", &setMatrix2DItem)
        .add_property("shape",       &getMatrix2DShape)
        .def_pickle(Matrix2DPickle());

    // std::vector<size_t> (actions...)
    class_<std::vector<size_t>>{"vec_size_t"}
        .def(vector_indexing_suite<std::vector<size_t>>());

    // Enable passing 3D tables from Python
    Vector3DFromPython<double>();
    Vector3DFromPython<int>();
}
