#ifndef AI_TOOLBOX_MDP_IO_HEADER_FILE
#define AI_TOOLBOX_MDP_IO_HEADER_FILE

#include <iostream>
#include <iomanip>

#include <AIToolbox/MDP/Policies/PolicyInterface.hpp>
#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/Model.hpp>

namespace AIToolbox::MDP {
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
     * @brief This function prints any MDP model to a file.
     *
     * @tparam M The type of the model.
     * @param os The output stream.
     * @param model The model to print.
     *
     * @return The resulting output stream.
     */
    // Here we use the =0 default template to avoid redefinition problems with other template ostream definitions.
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
    std::ostream& operator<<(std::ostream &os, const M & model) {
        const size_t S = model.getS();
        const size_t A = model.getA();

        for ( size_t s = 0; s < S; ++s ) {
            for ( size_t a = 0; a < A; ++a ) {
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    // The +2 is for first digit and the dot, since we know that here the max value possible is 1.0
                    os << std::setw(os.precision()+2) << std::left << model.getTransitionProbability(s, a, s1) << '\t'
                       << std::setw(os.precision()+2) << std::left << model.getExpectedReward(s, a, s1)        << '\t';
                }
            }
            os << '\n';
        }

        return os;
    }

    /**
     * @brief This function prints the whole policy to a stream.
     *
     * This function outputs each and every value of the policy
     * for easy parsing. The output is broken into multiple lines
     * where each line is of the format:
     *
     * state_number action_number probability
     *
     * And all lines are sorted by state, and each state is sorted
     * by action.
     *
     * @param os The stream where the policy is printed.
     * @param p The policy that is begin printed.
     *
     * @return The original stream.
     */
    std::ostream& operator<<(std::ostream &os, const PolicyInterface & p);

    /**
     * @brief This function prints an Experience to a stream.
     *
     * This function is able to print an Experience as long as it conforms
     * to the Experience interface, described in the is_experience struct.
     *
     * @tparam E The type of the Experience.
     * @param os The output stream.
     * @param model The Experience to print.
     *
     * @return The resulting output stream.
     */
    template <typename E, std::enable_if_t<is_experience_v<E>, int> = 0>
    std::ostream& operator<<(std::ostream &os, const E & exp) {
        const size_t S = exp.getS();
        const size_t A = exp.getA();

        for ( size_t s = 0; s < S; ++s ) {
            for ( size_t a = 0; a < A; ++a ) {
                // Visits, then rewards and M2s
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    os << exp.getVisits(s, a, s1) << ' ';
                }
                os << '\n' << exp.getReward(s, a) << ' ' << exp.getM2(s, a) << '\n';
            }
        }
        return os;
    }

    class Experience;
    /**
     * @brief This function implements input from stream for the MDP::Experience class.
     *
     * Note that as all other input function, it does not actually change the
     * input model if the reading fails.
     *
     * @param is The input stream.
     * @param e The Experience to be read.
     *
     * @return The original stream.
     */
    std::istream& operator>>(std::istream &is, Experience & e);

    class SparseExperience;
    /**
     * @brief This function implements input from stream for the MDP::SparseExperience class.
     *
     * Note that as all other input function, it does not actually change the
     * input model if the reading fails.
     *
     * @param is The input stream.
     * @param e The SparseExperience to be read.
     *
     * @return The original stream.
     */
    std::istream& operator>>(std::istream &is, SparseExperience & e);

    class Model;
    /**
     * @brief This function implements input from stream for the MDP::Model class.
     *
     * Note that as all other input function, it does not actually change the
     * input model if the reading fails.
     *
     * @param is The input stream.
     * @param m The model to write into.
     *
     * @return The input stream.
     */
    std::istream& operator>>(std::istream &is, Model & m);

    class Policy;
    /**
     * @brief This function implements input from stream for the MDP::Model class.
     *
     * Note that as all other input function, it does not actually change the
     * input model if the reading fails.
     *
     * @param is The input stream.
     * @param p The policy to write into.
     *
     * @return The input stream.
     */
    std::istream& operator>>(std::istream &is, Policy & p);
}

#endif
