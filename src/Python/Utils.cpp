#include "Utils.hpp"

#include <AIToolbox/Types.hpp>

EigenVectorFromPython::EigenVectorFromPython() {
    boost::python::converter::registry::push_back(
        &EigenVectorFromPython::convertible,
        &EigenVectorFromPython::construct,
        boost::python::type_id<AIToolbox::Vector>(
    ));
}

void* EigenVectorFromPython::convertible(PyObject* obj_ptr) {
    if (!PyList_Check(obj_ptr)) return 0;
    return obj_ptr;
}

void EigenVectorFromPython::construct(PyObject* list, boost::python::converter::rvalue_from_python_stage1_data* data)
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
