#ifndef AI_TOOLBOX_MDP_IO_HEADER_FILE
#define AI_TOOLBOX_MDP_IO_HEADER_FILE

#include <iosfwd>

namespace AIToolbox::MDP {
    // Forward references to avoid including tons of headers
    class Experience;
    class SparseExperience;
    class Model;
    class SparseModel;
    class PolicyInterface;
    class Policy;

    /**
     * @brief This function parses an MDP from a Cassandra formatted stream.
     *
     * This function may throw std::runtime_errors depending on whether the
     * input is correctly formed or not.
     *
     * @param input The input stream.
     *
     * @return The parsed model.
     */
    Model parseCassandra(std::istream & input);

    /**
     * @name MDP output stream operators.
     *
     * These utilities output MDP types at the maximum possible precision.
     * This allows reading them back with no loss of accuracy.
     *
     * @{
     */

    std::ostream & operator<<(std::ostream & os, const Model & model);
    std::ostream & operator<<(std::ostream & os, const SparseModel & model);
    std::ostream & operator<<(std::ostream & os, const Experience & exp);
    std::ostream & operator<<(std::ostream & os, const SparseExperience & exp);
    std::ostream & operator<<(std::ostream & os, const PolicyInterface & p);

    /** @}  */

    /**
     * @name Input stream utilities
     *
     * These utilities read back data outputted with their respective
     * operator<<() function.
     *
     * Note that the inputs must already be constructed with the correct size
     * (state-action spaces), as the operator<<() do not save this information.
     *
     * These functions do not modify the input if the parsing fails.
     *
     * @{
     */

    std::istream& operator>>(std::istream &is, Model & m);
    std::istream& operator>>(std::istream &is, SparseModel & m);
    std::istream& operator>>(std::istream &is, Experience & e);
    std::istream& operator>>(std::istream &is, SparseExperience & e);
    std::istream& operator>>(std::istream &is, Policy & p);

    /** @}  */
}

#endif
