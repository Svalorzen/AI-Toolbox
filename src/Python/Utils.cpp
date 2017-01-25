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

void exportUtils() {
    using namespace AIToolbox;
    using namespace boost::python;

    // Results of POMDP policy with horizon
    TupleToPython<std::tuple<size_t, size_t>>();
    // Results of sampleSR
    TupleToPython<std::tuple<size_t, double>>();
    // Results of sampleSOR
    TupleToPython<std::tuple<size_t, size_t, double>>();
    // General values (value function, POMDP belief..)
    class_<Vector>{"Vector", init<int>()}
        .def("__getitem__", &getVectorItem)
        .def("__setitem__", &setVectorItem)
        .def("__len__",     &getVectorLen);

    // Actions
    class_<std::vector<size_t>>{"vec_size_t"}
        .def(vector_indexing_suite<std::vector<size_t>>());
    // MDP Value Function
    TupleToPython<std::tuple<AIToolbox::Vector, std::vector<size_t>>>();
    TupleFromPython<AIToolbox::Vector, std::vector<size_t>>();
    // MDP QFunction
    class_<Matrix2D>{"Matrix2D", init<int, int>()}
        .def("__getitem__", &getMatrix2DItem)
        .def("__setitem__", &setMatrix2DItem)
        .add_property("shape",       &getMatrix2DShape);

    // Enable passing starting value functions from Python
    VectorFromPython<double>();
    // Enable passing 3D transition/reward tables from Python to MDP::Model
    Vector3DFromPython<double>();
    Vector3DFromPython<int>();

    EigenVectorFromPython();

    // POMDP Value Function (VEntry)
    using VEntry = std::tuple<AIToolbox::Vector, size_t, std::vector<size_t>>;
    using VList = std::vector<VEntry>;
    TupleToPython<VEntry>();
    class_<std::vector<VEntry>>{"VList"}
        .def(vector_indexing_suite<std::vector<VEntry>>());
    class_<std::vector<VList>>{"VFun"}
        .def(vector_indexing_suite<std::vector<VList>>());
    TupleToPython<std::tuple<bool, std::vector<VList>>>();
}
