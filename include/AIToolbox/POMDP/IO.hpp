#ifndef AI_TOOLBOX_POMDP_IO_HEADER_FILE
#define AI_TOOLBOX_POMDP_IO_HEADER_FILE

#include <iostream>
#include <iomanip>

#include <AIToolbox/MDP/IO.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>

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
     * @brief This function prints any POMDP model to a file.
     *
     * @tparam M The type of the model.
     * @param os The output stream.
     * @param model The model to print.
     *
     * @return The resulting output stream.
     */
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
    std::ostream& operator<<(std::ostream &os, const M & model) {
        // First print the MDP part
        MDP::operator<<(os, model);

        const size_t S = model.getS();
        const size_t A = model.getA();
        const size_t O = model.getO();

        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                for ( size_t o = 0; o < O; ++o ) {
                    // The +2 is for first digit and the dot, since we know that here the max value possible is 1.0
                    os << std::setw(os.precision()+2) << std::left << model.getObservationProbability(s1, a, o) << '\t';
                }
            }
            os << '\n';
        }

        return os;
    }

    /**
     * @brief This function implements input from stream for the POMDP::Model class.
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
    template <typename M, typename>
    std::istream& operator>>(std::istream &is, Model<M> & m) {
        const size_t S = m.getS();
        const size_t A = m.getA();
        const size_t O = m.getO();

        Model<M> in(O,S,A);
        MDP::operator>>(is, in);

        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                double sum = 0.0;
                for ( size_t o = 0; o < O; ++o ) {
                    if ( !(is >> in.observations_[a](s1, o))) {
                        std::cerr << "AIToolbox: Could not read Model data.\n";
                        is.setstate(std::ios::failbit);
                        return is;
                    }
                    sum += in.observations_[a](s1, o);
                }

                if ( checkDifferentSmall(sum, 0.0) )
                    in.observations_[a].row(s1) /= sum;
                else
                    in.observations_[a](s1, 0) = 1.0;
            }
        }
        // This guarantees that if input is invalid we still keep the old Model.
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
    template <typename M, typename>
    std::istream& operator>>(std::istream &is, SparseModel<M> & m) {
        const size_t S = m.getS();
        const size_t A = m.getA();
        const size_t O = m.getO();

        SparseModel<M> in(O,S,A);
        MDP::operator>>(is, in);

        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                double sum = 0.0;
                for ( size_t o = 0; o < O; ++o ) {
                    double p;
                    if ( !(is >> p) ) {
                        std::cerr << "AIToolbox: Could not read Model data.\n";
                        is.setstate(std::ios::failbit);
                        return is;
                    }
                    if ( checkDifferentSmall(p, 0.0) ) {
                        in.observations_[a].coeffRef(s1, o) = p;
                        sum += p;
                    }
                }

                if ( checkDifferentSmall(sum, 0.0) )
                    in.observations_[a].row(s1) /= sum;
                else
                    in.observations_[a].coeffRef(s1, 0) = 1.0;
            }
        }
        // This guarantees that if input is invalid we still keep the old Model.
        m = std::move(in);

        return is;
    }

    /**
     * @brief This function reads a policy from a file.
     *
     * This function reads files that have been outputted through
     * operator<<(). If not enough values can be extracted from
     * the stream, the function stops and the input policy is
     * not modified. In addition, it checks whether the probability
     * values are within 0 and 1.
     *
     * @param is The stream were the policy is being read from.
     * @param p The policy that is being assigned.
     *
     * @return The input stream.
     */
    std::istream& operator>>(std::istream &is, Policy & p);

    /**
     * @brief This function prints the whole policy to a stream.
     *
     * This function basically outputs the internal ValueFunction
     * in a recoverable format.
     *
     * @param os The stream where the policy is printed.
     * @param p The policy that is begin printed.
     *
     * @return The original stream.
     */
    std::ostream& operator<<(std::ostream &os, const Policy & p);
}

#endif
