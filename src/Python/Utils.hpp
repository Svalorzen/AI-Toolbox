#ifndef AI_TOOLBOX_PYTHON_UTILS_HEADER_FILE
#define AI_TOOLBOX_PYTHON_UTILS_HEADER_FILE

#include <cstddef>
#include <vector>
#include <tuple>

#include <boost/python.hpp>

// C++ to Python

template <typename T>
struct TupleToPython {
    TupleToPython() {
        boost::python::to_python_converter<T, TupleToPython<T>>();
    }

    template<int...>
    struct sequence {};

    template<int N, int... S>
    struct generator : generator<N-1, N-1, S...> { };

    template<int... S>
    struct generator<0, S...> {
        using type = sequence<S...>;
    };

    template <int... I>
    static boost::python::tuple boostConvertImpl(const T& t, sequence<I...>) {
        return boost::python::make_tuple(std::get<I>(t)...);
    }

    static boost::python::tuple boostConvert(const T & t) {
        return boostConvertImpl(t, typename generator<std::tuple_size_v<T>>::type());
    }

    static PyObject* convert(const T& t) {
        return boost::python::incref(boostConvert(t).ptr());
    }
};

template <typename T>
struct PairToPython {
    PairToPython() {
        boost::python::to_python_converter<T, PairToPython<T>>();
    }

    static PyObject* convert(const T& t) {
        return boost::python::incref(boost::python::make_tuple(t.first, t.second).ptr());
    }
};

// Python to C++

template <typename T>
struct TupleFromPython {
    TupleFromPython() {
        boost::python::converter::registry::push_back(&TupleFromPython::convertible, &TupleFromPython::construct, boost::python::type_id<T>());
    }

    static void* convertible(PyObject* obj_ptr) {
        if (!PyTuple_CheckExact(obj_ptr)) return 0;
        return obj_ptr;
    }

    template <size_t Id, bool = true>
    struct ExtractPythonTuple {
        void operator()(T & t, PyObject * tuple) {
            std::get<Id>(t) = boost::python::extract<std::tuple_element_t<Id, T>>(PyTuple_GetItem(tuple, Id));
            ExtractPythonTuple<Id - 1>()(t, tuple);
        }
    };

    template <bool dummyForSpecialization>
    struct ExtractPythonTuple<0, dummyForSpecialization> {
        void operator()(T & t, PyObject * tuple) {
            std::get<0>(t) = boost::python::extract<std::tuple_element_t<0, T>>(PyTuple_GetItem(tuple, 0));
        }
    };

    static void construct(PyObject* tuple, boost::python::converter::rvalue_from_python_stage1_data* data) {
        // Grab pointer to memory into which to construct the new tuple
        void* storage = ((boost::python::converter::rvalue_from_python_storage<T>*)data)->storage.bytes;

        T& t = *(new (storage) T());

        // Copy item by item the tuple
        ExtractPythonTuple<std::tuple_size_v<T> - 1>()(t, tuple);

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
};

template <typename T>
struct PairFromPython {
    PairFromPython() {
        boost::python::converter::registry::push_back(&PairFromPython::convertible, &PairFromPython::construct, boost::python::type_id<T>());
    }

    static void* convertible(PyObject* obj_ptr) {
        if (!PyTuple_CheckExact(obj_ptr)) return 0;
        return obj_ptr;
    }

    static void construct(PyObject* tuple, boost::python::converter::rvalue_from_python_stage1_data* data) {
        // Grab pointer to memory into which to construct the new tuple
        void* storage = ((boost::python::converter::rvalue_from_python_storage<T>*)data)->storage.bytes;

        T& t = *(new (storage) T());

        // Copy the pair in
        t.first  = boost::python::extract<std::tuple_element_t<0, T>>(PyTuple_GetItem(tuple, 0));
        t.second = boost::python::extract<std::tuple_element_t<1, T>>(PyTuple_GetItem(tuple, 1));

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
};

struct EigenVectorFromPython {
    EigenVectorFromPython();

    static void* convertible(PyObject* obj_ptr);
    static void construct(PyObject* list, boost::python::converter::rvalue_from_python_stage1_data* data);
};

template<typename T>
struct VectorFromPython {
    VectorFromPython() {
        boost::python::converter::registry::push_back(&VectorFromPython<T>::convertible, &VectorFromPython<T>::construct, boost::python::type_id<std::vector<T>>());
    }

    static void* convertible(PyObject* obj_ptr) {
        if (!PyList_Check(obj_ptr)) return 0;
        return obj_ptr;
    }

    static void construct(PyObject* list, boost::python::converter::rvalue_from_python_stage1_data* data) {
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

template <typename T>
struct Vector3DFromPython {
    using V3D = std::vector<std::vector<std::vector<T>>>;

    Vector3DFromPython() {
        boost::python::converter::registry::push_back(&Vector3DFromPython<T>::convertible, &Vector3DFromPython<T>::construct, boost::python::type_id<V3D>());
    }

    static void* convertible(PyObject* obj_ptr) {
        if (!PyList_Check(obj_ptr) ||
            !PyList_Check(PyList_GetItem(obj_ptr,0)) ||
            !PyList_Check(PyList_GetItem(PyList_GetItem(obj_ptr,0),0))) return 0;
        return obj_ptr;
    }

    static void construct(PyObject* list, boost::python::converter::rvalue_from_python_stage1_data* data) {
        // Grab pointer to memory into which to construct the new std::vector<T>
        void* storage = ((boost::python::converter::rvalue_from_python_storage<V3D>*)data)->storage.bytes;

        V3D& v = *(new (storage) V3D());

        // Copy item by item the list
        auto size3 = PyList_Size(list);
        v.resize(size3);
        for(decltype(size3) i = 0; i < size3; ++i) {
            auto size2 = PyList_Size(PyList_GetItem(list,0));
            v[i].resize(size2);
            for(decltype(size2) j = 0; j < size2; ++j) {
                auto size1 = PyList_Size(PyList_GetItem(PyList_GetItem(list,0),0));
                v[i][j].resize(size1);
                for(decltype(size1) k = 0; k < size1; ++k)
                    v[i][j][k] = boost::python::extract<T>(PyList_GetItem(PyList_GetItem(PyList_GetItem(list, i), j), k));
            }
        }

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
};

#endif
