#ifndef AI_TOOLBOX_POMDP_IO_HEADER_FILE
#define AI_TOOLBOX_POMDP_IO_HEADER_FILE

#include <iostream>
#include <iomanip>

#include <AIToolbox/MDP/IO.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This function prints any POMDP model to a file.
     *
     * @tparam M The type of the model.
     * @param os The output stream.
     * @param model The model to print.
     *
     * @return The resulting output stream.
     */
    template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
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
}

#endif
