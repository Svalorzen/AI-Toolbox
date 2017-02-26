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

void exportTypes() {
    using namespace AIToolbox;
    using namespace boost::python;

    // Eigen Vector
    class_<Vector>{"Vector", init<int>()}
        .def("__getitem__", &getVectorItem)
        .def("__setitem__", &setVectorItem)
        .def("__len__",     &getVectorLen);

    EigenVectorFromPython();

    // 2D Eigen matrix
    class_<Matrix2D>{"Matrix2D", init<int, int>()}
        .def("__getitem__", &getMatrix2DItem)
        .def("__setitem__", &setMatrix2DItem)
        .add_property("shape",       &getMatrix2DShape);

    // std::vector<size_t> (actions...)
    class_<std::vector<size_t>>{"vec_size_t"}
        .def(vector_indexing_suite<std::vector<size_t>>());

    // Enable passing 3D tables from Python
    Vector3DFromPython<double>();
    Vector3DFromPython<int>();
}
