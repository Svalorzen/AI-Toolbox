#ifndef AI_TOOLBOX_PYTHON_UTILS_HEADER_FILE
#define AI_TOOLBOX_PYTHON_UTILS_HEADER_FILE

#include <cstddef>

#include <AIToolbox/Types.hpp>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

// C++ to Python

template <typename T>
struct TupleToPython {
    TupleToPython() {
        boost::python::to_python_converter<T, TupleToPython<T>>();
    }

    template<int ...>
    struct sequence {};

    template<int N, int ...S>
    struct generator : generator<N-1, N-1, S...> { };

    template<int ...S>
    struct generator<0, S...> {
        using type = sequence<S...>;
    };

    template <int... I>
    static boost::python::tuple boostConvertImpl(const T& t, sequence<I...>) {
        return boost::python::make_tuple(std::get<I>(t)...);
    }

    template <typename... Args>
    static boost::python::tuple boostConvert(const std::tuple<Args...> & t) {
        return boostConvertImpl(t, typename generator<sizeof...(Args)>::type());
    }

    static PyObject* convert(const T& t) {
        return boost::python::incref(boostConvert(t).ptr());
    }
};

// Python to C++

template<typename T>
struct VectorFromPython {
    VectorFromPython() {
        boost::python::converter::registry::push_back(&VectorFromPython<T>::convertible, &VectorFromPython<T>::construct, boost::python::type_id<std::vector<T>>());
    }

    static void* convertible(PyObject* obj_ptr) {
        if (!PyList_Check(obj_ptr)) return 0;
        return obj_ptr;
    }

    static void construct(PyObject* list, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        // Grab pointer to memory into which to construct the new std::vector<T>
        void* storage = ((boost::python::converter::rvalue_from_python_storage<std::vector<T>>*)data)->storage.bytes;

        std::vector<T>& v = *(new (storage) std::vector<T>());

        // Copy item by item the list
        auto size = PyList_Size(list);
        v.resize(size);
        for(decltype(size) i = 0; i < size; ++i)
            v[i] = boost::python::extract<T>(PyList_GetItem(list, i));

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
};

// Util classes

double getVectorItem(const AIToolbox::Vector& v, int index) {
    return v(index);
}

void setVectorItem(AIToolbox::Vector& v, int index, double value) {
    v(index) = value;
}

double getMatrix2DItem(const AIToolbox::Matrix2D& m, boost::python::tuple i) {
    return m(boost::python::extract<int>(i[0]), boost::python::extract<int>(i[1]));
}

void setMatrix2DItem(AIToolbox::Matrix2D & m, boost::python::tuple i, double value) {
    m(boost::python::extract<int>(i[0]), boost::python::extract<int>(i[1])) = value;
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
        .def("__setitem__", &setVectorItem);

    // Actions
    class_<std::vector<size_t>>{"vec_size_t"}
        .def(vector_indexing_suite<std::vector<size_t>>());
    // MDP Value Function
    TupleToPython<std::tuple<AIToolbox::Vector, std::vector<size_t>>>();
    // MDP QFunction
    class_<Matrix2D>{"Matrix2D", init<int, int>()}
        .def("__getitem__", &getMatrix2DItem)
        .def("__setitem__", &setMatrix2DItem);

    // Enable passing starting value functions from Python
    VectorFromPython<double>();
    // Enable passing 3D transition/reward tables from Python to MDP::Model
    VectorFromPython<std::vector<std::vector<double>>>();
}

#endif
