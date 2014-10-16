#ifndef AI_TOOLBOX_POMDP_IO_HEADER_FILE
#define AI_TOOLBOX_POMDP_IO_HEADER_FILE

#include <iostream>
#include <iomanip>

#include <AIToolbox/MDP/IO.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Model.hpp>

namespace AIToolbox {
    namespace POMDP {
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

            size_t S = model.getS();
            size_t A = model.getA();
            size_t O = model.getO();

            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                for ( size_t a = 0; a < A; ++a ) {
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
            size_t S = m.getS();
            size_t A = m.getA();
            size_t O = m.getO();

            Model<M> in(O,S,A);
            MDP::operator>>(is, in);

            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                for ( size_t a = 0; a < A; ++a ) {
                    for ( size_t o = 0; o < O; ++o ) {
                        is >> in.observations_[s1][a][o];
                    }
                    // Verification/Sanitization
                    auto ref = in.observations_[s1][a];
                    normalizeProbability(std::begin(ref), std::end(ref), std::begin(ref));
                }
            }
            // This guarantees that if input is invalid we still keep the old Model.
            m = in;

            return is;
        }
    }
}
#endif
