#ifndef AI_TOOLBOX_PYTHON_UTILS_HEADER_FILE
#define AI_TOOLBOX_PYTHON_UTILS_HEADER_FILE

#include <cstddef>
#include <vector>

#include <boost/python.hpp>
#include <AIToolbox/Types.hpp>

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

struct EigenVectorFromPython {
    EigenVectorFromPython() {
        boost::python::converter::registry::push_back(&EigenVectorFromPython::convertible, &EigenVectorFromPython::construct, boost::python::type_id<AIToolbox::Vector>());
    }

    static void* convertible(PyObject* obj_ptr) {
        if (!PyList_Check(obj_ptr)) return 0;
        return obj_ptr;
    }

    static void construct(PyObject* list, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        // Grab pointer to memory into which to construct the new Vector
        void* storage = ((boost::python::converter::rvalue_from_python_storage<AIToolbox::Vector>*)data)->storage.bytes;

        AIToolbox::Vector& v = *(new (storage) AIToolbox::Vector());

        // Copy item by item the list
        auto size = PyList_Size(list);
        v.resize(size);
        for(decltype(size) i = 0; i < size; ++i)
            v[i] = boost::python::extract<double>(PyList_GetItem(list, i));

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
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

    static void construct(PyObject* list, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
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
