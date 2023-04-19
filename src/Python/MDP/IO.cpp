#include <AIToolbox/MDP/IO.hpp>

#include <boost/python.hpp>

#include <string>
#include <sstream>

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>

void exportMDPIO() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    struct MDPIOWrapper {
        std::string write(const Model & m) const {
            std::stringstream out;
            out << m;
            return out.str();
        }

        std::string write(const SparseModel & m) const {
            std::stringstream out;
            out << m;
            return out.str();
        }

        std::string write(const Experience & e) const {
            std::stringstream out;
            out << e;
            return out.str();
        }

        std::string write(const SparseExperience & e) const {
            std::stringstream out;
            out << e;
            return out.str();
        }

        void read(const std::string & s, Model & m) const {
            std::stringstream in(s);
            in >> m;
        }

        void read(const std::string & s, SparseModel & m) const {
            std::stringstream in(s);
            in >> m;
        }

        void read(const std::string & s, Experience & e) const {
            std::stringstream in(s);
            in >> e;
        }

        void read(const std::string & s, SparseExperience & e) const {
            std::stringstream in(s);
            in >> e;
        }
    };

    class_<MDPIOWrapper>{"IO",

              "This class wraps C++ MDP IO functionality.\n"
              "\n"
              "While the models in Python can be pickled in order to save them,\n"
              "this does not allow direct interaction between C++ and Python code.\n"
              "\n"
              "This class wraps the common AIToolbox operator<< and operator>> for\n"
              "MDP classes, so that they can be saved and loaded equally from both\n"
              "C++ and Python. The format is human-friendly (and thus\n"
              "space-unfriendly); if the models are supposed to only be used in\n"
              "Python, pickling is recommended.\n"
              "\n"
              "The models are returned in strings as to avoid having Boost Python\n"
              "pass Python files to C++; the strings can be saved on files in\n"
              "whatever method you wish"}

        .def("writeModel", static_cast<std::string(MDPIOWrapper::*)(const Model&) const>(&MDPIOWrapper::write))
        .def("writeSparseModel", static_cast<std::string(MDPIOWrapper::*)(const SparseModel&) const>(&MDPIOWrapper::write))
        .def("writeExperience", static_cast<std::string(MDPIOWrapper::*)(const Experience&) const>(&MDPIOWrapper::write))
        .def("writeSparseExperience", static_cast<std::string(MDPIOWrapper::*)(const SparseExperience&) const>(&MDPIOWrapper::write))
        .def("readModel", static_cast<void(MDPIOWrapper::*)(const std::string &, Model&) const>(&MDPIOWrapper::read))
        .def("readSparseModel", static_cast<void(MDPIOWrapper::*)(const std::string &, SparseModel&) const>(&MDPIOWrapper::read))
        .def("readExperience", static_cast<void(MDPIOWrapper::*)(const std::string &, Experience&) const>(&MDPIOWrapper::read))
        .def("readSparseExperience", static_cast<void(MDPIOWrapper::*)(const std::string &, SparseExperience&) const>(&MDPIOWrapper::read));
}
