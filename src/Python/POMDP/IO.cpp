#include <AIToolbox/POMDP/IO.hpp>

#include <boost/python.hpp>

#include <string>
#include <sstream>

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

using POMDPModelBinded = AIToolbox::POMDP::Model<AIToolbox::MDP::Model>;
using POMDPSparseModelBinded = AIToolbox::POMDP::SparseModel<AIToolbox::MDP::SparseModel>;

void exportPOMDPIO() {
    using namespace AIToolbox::POMDP;
    using namespace boost::python;

    struct POMDPIOWrapper {
        std::string write(const POMDPModelBinded & m) const {
            std::stringstream out;
            out << m;
            return out.str();
        }

        std::string write(const POMDPSparseModelBinded & m) const {
            std::stringstream out;
            out << m;
            return out.str();
        }

        void read(const std::string & s, POMDPModelBinded & m) const {
            std::stringstream in(s);
            in >> m;
        }

        void read(const std::string & s, POMDPSparseModelBinded & m) const {
            std::stringstream in(s);
            in >> m;
        }
    };

    class_<POMDPIOWrapper>{"IO",

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

        .def("writeModel", static_cast<std::string(POMDPIOWrapper::*)(const POMDPModelBinded&) const>(&POMDPIOWrapper::write))
        .def("writeSparseModel", static_cast<std::string(POMDPIOWrapper::*)(const POMDPSparseModelBinded&) const>(&POMDPIOWrapper::write))
        .def("readModel", static_cast<void(POMDPIOWrapper::*)(const std::string &, POMDPModelBinded&) const>(&POMDPIOWrapper::read))
        .def("readSparseModel", static_cast<void(POMDPIOWrapper::*)(const std::string &, POMDPSparseModelBinded&) const>(&POMDPIOWrapper::read));
}
