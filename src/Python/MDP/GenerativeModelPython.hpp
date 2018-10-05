#ifndef AI_TOOLBOX_MDP_GENERATIVE_MODEL_PYTHON_HEADER_FILE
#define AI_TOOLBOX_MDP_GENERATIVE_MODEL_PYTHON_HEADER_FILE

#include <boost/python.hpp>
#include <boost/python/object.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class allows to import generative models from Python.
     *
     * This class wraps an instance of a Python class that provides generator
     * methods to sample states and rewards from, so that one does not need to
     * always specify transition and reward functions from Python.
     */
    class GenerativeModelPython {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor takes a Python object, which will be used to
             * call the generative methods from C++.
             *
             * This class expects the instance to have at least the following methods:
             *
             * - getS(): returns the number of states of the environment.
             * - getA(): returns the number of actions of the environment, in ALL states.
             * - getDiscount(): returns the discount of the environment, [0, 1].
             * - isTerminal(s): returns whether a given state is a terminal state.
             * - sampleSR(s, a): returns a tuple containing new state and reward, from the input state and action.
             *
             * @param instance The Python object instance to call methods on.
             */
            GenerativeModelPython(boost::python::object instance) :
                    instance_(instance) {}

            /**
             * @brief This function returns the number of states of the environment.
             */
            size_t getS() const { return boost::python::extract<size_t>(instance_.attr("getS")()); }

            /**
             * @brief This function returns the number of actions of the environment.
             */
            size_t getA() const { return boost::python::extract<size_t>(instance_.attr("getA")()); }

            /**
             * @brief This function returns the discount of the environment.
             */
            double getDiscount() const { return boost::python::extract<double>(instance_.attr("getDiscount")()); }

            /**
             * @brief This function returns whether a given state is a terminal state.
             */
            bool isTerminal(const size_t s) const { return boost::python::extract<bool>(instance_.attr("isTerminal")(s)); }

            /**
             * @brief This function returns a tuple containing a new state and reward, from the input state and action.
             */
            std::tuple<size_t, double> sampleSR(const size_t s, const size_t a) const {
                return boost::python::extract<std::tuple<size_t, double>>(instance_.attr("sampleSR")(s, a));
            }

        private:
            boost::python::object instance_;
    };
}

#endif
