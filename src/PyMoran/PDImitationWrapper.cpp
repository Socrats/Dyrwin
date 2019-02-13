//
// Created by Elias Fernandez on 2019-02-11.
//

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <vector>
#include "PDImitation.h"

namespace py = boost::python;
using namespace egt_tools;

template<typename T>
struct VectorFromPython {
    VectorFromPython() {
        py::converter::registry::push_back(&VectorFromPython<T>::convertible,
                                           &VectorFromPython<T>::construct,
                                           py::type_id<std::vector<T>>());
    }

    static void *convertible(PyObject *obj_ptr) {
        if (!PyList_Check(obj_ptr)) return 0;
        return obj_ptr;
    }

    static void construct(PyObject *list, py::converter::rvalue_from_python_stage1_data *data) {
        // Grab pointer to memory into which to construct the new std::vector<T>
        void *storage = ((py::converter::rvalue_from_python_storage<std::vector<T>> *) data)->storage.bytes;

        std::vector<T> &v = *(new(storage) std::vector<T>());

        // Copy item by item the list
        auto size = PyList_Size(list);
        v.resize(static_cast<unsigned long>(size));
        for (decltype(size) i = 0; i < size; ++i)
            v[i] = py::extract<T>(PyList_GetItem(list, i));

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
};

BOOST_PYTHON_MODULE (EGTtools) {

    py::class_<std::vector<float>>{"vec_float"}
            .def(py::vector_indexing_suite<std::vector<float>>());
    VectorFromPython<float>();

    py::class_<PDImitation>("PDImitation",
                            py::init<unsigned int, unsigned int, float, float, float, std::vector<float>>())
            .add_property("generations", &PDImitation::generations, &PDImitation::set_generations)
            .add_property("pop_size", &PDImitation::pop_size, &PDImitation::set_pop_size)
            .add_property("nb_coop", &PDImitation::nb_coop)
            .add_property("mu", &PDImitation::mu, &PDImitation::set_mu)
            .add_property("beta", &PDImitation::beta, &PDImitation::set_beta)
            .add_property("coop_freq", &PDImitation::coop_freq, &PDImitation::set_coop_freq)
            .add_property("result_coop_freq", &PDImitation::result_coop_freq)
            .add_property("payoff_matrix", &PDImitation::payoff_matrix, &PDImitation::set_payoff_matrix)
            .def("evolve", static_cast<float (PDImitation::*)(float)>(&PDImitation::evolve))
            .def("evolve", static_cast<float (PDImitation::*)(unsigned int, float)>(&PDImitation::evolve))
            .def("evolve",
                 static_cast<std::vector<float> (PDImitation::*)(std::vector<float>, unsigned int) >
                 (&PDImitation::evolve));
}