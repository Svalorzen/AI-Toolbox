#ifndef AI_TOOLBOX_POMDP_IO_HEADER_FILE
#define AI_TOOLBOX_POMDP_IO_HEADER_FILE

#include <iostream>
#include <iomanip>

#include <AIToolbox/Utils/IO.hpp>
#include <AIToolbox/MDP/IO.hpp>

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>

#include <AIToolbox/Logging.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This function parses a POMDP from a Cassandra formatted stream.
     *
     * This function may throw std::runtime_errors depending on whether the
     * input is correctly formed or not.
     *
     * @param input The input stream.
     *
     * @return The parsed model.
     */
    Model<MDP::Model> parseCassandra(std::istream & input);

    /**
     * @brief This function outputs a POMDP model to a stream.
     *
     * @tparam M The type of the model.
     * @param os The output stream.
     * @param model The model to output.
     *
     * @return The resulting output stream.
     */
    template <IsModelEigen M>
    std::ostream& operator<<(std::ostream &os, const M & model) {
        // First print the MDP part
        MDP::operator<<(os, model);

        write(os, model.getObservationFunction());
        return os;
    }

    /**
     * @brief This function parses a Model from a stream.
     *
     * This function does not modify the input model if the parsing fails.
     *
     * @tparam M The underlying MDP model. Needs to have operator>> implemented.
     * @param is The input stream.
     * @param m The model to write into.
     *
     * @return The input stream.
     */
    template <MDP::IsModel M>
    std::istream& operator>>(std::istream &is, Model<M> & m) {
        Model<M> in(m.getO(), m.getS(), m.getA());

        if (!MDP::operator>>(is, in)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read underlying MDP for POMDP Model.");
            return is;
        }

        auto observations = in.getObservationFunction();
        if (!read(is, observations)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read Model<M> observation function.");
            return is;
        } else {
            try {
                in.setObservationFunction(observations);
            } catch (const std::invalid_argument &) {
                AI_LOGGER(AI_SEVERITY_ERROR, "The observation function for Model<M> did not contain valid probabilities.");
                is.setstate(std::ios::failbit);
                return is;
            }
        }

        m = std::move(in);

        return is;
    }

    /**
     * @brief This function implements input from stream for the POMDP::SparseModel class.
     *
     * Note that as all other input function, it does not actually change the
     * input model if the reading fails.
     *
     * @tparam M The underlying MDP model. Needs to have operator<< implemented.
     * @param is The input stream.
     * @param m The model to write into.
     *
     * @return The input stream.
     */
    template <MDP::IsModel M>
    std::istream& operator>>(std::istream &is, SparseModel<M> & m) {
        SparseModel<M> in(m.getO(), m.getS(), m.getA());

        if (!MDP::operator>>(is, in)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read underlying MDP for POMDP Model.");
            return is;
        }

        auto observations = in.getObservationFunction();
        if (!read(is, observations)) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Could not read SparseModel<M> observation function.");
            return is;
        } else {
            try {
                in.setObservationFunction(observations);
            } catch (const std::invalid_argument &) {
                AI_LOGGER(AI_SEVERITY_ERROR, "The observation function for SparseModel<M> did not contain valid probabilities.");
                is.setstate(std::ios::failbit);
                return is;
            }
        }

        m = std::move(in);

        return is;
    }

    /**
     * @brief This function outputs a Policy to a stream.
     *
     * @param os The stream where the policy is printed.
     * @param p The policy that is begin printed.
     *
     * @return The original stream.
     */
    std::ostream& operator<<(std::ostream &os, const Policy & p);

    /**
     * @brief This function reads a policy from a file.
     *
     * This function reads files that have been outputted through
     * operator<<(std::ostream&, const Policy&). If not enough values can be
     * extracted from the stream, the function stops and the input policy is
     * not modified. In addition, it checks whether the probability values are
     * within 0 and 1.
     *
     * @param is The stream were the policy is being read from.
     * @param p The policy that is being assigned.
     *
     * @return The input stream.
     */
    std::istream& operator>>(std::istream &is, Policy & p);
}

#endif
