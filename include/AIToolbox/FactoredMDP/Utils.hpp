#ifndef AI_TOOLBOX_FACTORED_MDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_UTILS_HEADER_FILE

#include <AIToolbox/FactoredMDP/Types.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        /**
         * @brief This function returns whether the common factors in the inputs match in value.
         *
         * @param lhs The left hand side.
         * @param rhs The right hand side.
         *
         * @return True if all factors in common between the inputs match in value, false otherwise.
         */
        bool match(const PartialFactors & lhs, const PartialFactors & rhs);

        /**
         * @brief This function appends the rhs to the lhs, assuming the original Factor for lhs has S elements.
         *
         * @param S The number of factors the full Factor for lhs has.
         * @param lhs The left hand side that gets extended in place.
         * @param rhs The right hand side.
         */
        void join(size_t S, PartialFactors * lhs, const PartialFactors & rhs);

        /**
         * @brief This function creates a new PartialFactors appending rhs to lhs, assuming lhs has S elements.
         *
         * @param S The number of factors the full Factor for lhs has.
         * @param lhs The left hand side.
         * @param rhs The right hand side.
         *
         * @return A new PartialFactors that contains all elements from lhs, and all elements from rhs shifted by S.
         */
        PartialFactors join(size_t S, const PartialFactors & lhs, const PartialFactors & rhs);

        /**
         * @brief This function creates a new Factor appending rhs to lhs.
         *
         * @param lhs The left hand side.
         * @param rhs The right hand side.
         *
         * @return A new Factor containing all elements from lhs and rhs.
         */
        Factors join(const Factors & lhs, const Factors & rhs);

        /**
         * @brief This function merges two PartialFactors together.
         *
         * This function assumes that all elements in the factors have
         * different keys. If they have the same keys, the behaviour is
         * unspecified.
         *
         * @param lhs The left hand side.
         * @param rhs The right hand side.
         *
         * @return A new PartialFactors containing all keys from both inputs and their respective values.
         */
        PartialFactors merge(const PartialFactors & lhs, const PartialFactors & rhs);

        /**
         * @brief This function merges the second PartialFactors into the first.
         *
         * This function assumes that all elements in the factors have
         * different keys. If they have the same keys, the behaviour is
         * unspecified.
         *
         * @param lhs The left hand side to be modified.
         * @param rhs The right hand side.
         */
        void inplace_merge(PartialFactors * plhs, const PartialFactors & rhs);

        /**
         * @brief This function returns the multiplication of all elements of the input factor.
         *
         * In case the factor space is too big to represent via a size_t, the
         * maximum possible representable value is returned.
         *
         * @param f The factored factor space.
         *
         * @return The possible number of factors if representable, otherwise the max size_t.
         */
        size_t factorSpace(const Factors & f);

        /**
         * @brief This function converts Factors into the equivalent PartialFactors structure.
         *
         * @param f The Factors to be converted.
         *
         * @return A PartialFactors structure equivalent to the input.
         */
        PartialFactors toPartialFactors(const Factors & f);

        /**
         * @brief This class enumerates all possible values for a PartialFactors.
         *
         * This class is a simple enumerator that goes through all possible
         * values of a PartialFactors for the specific input factors. An
         * additional separate factor index can be specified in order to skip
         * that factor, to allow the user to modify that freely.
         */
        class PartialFactorsEnumerator {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * This constructor initializes the internal PartialFactors
                 * with the factors obtained as inputs. In addition it saves
                 * the input Factor as the ceiling for the values in the
                 * PartialFactors.
                 *
                 * @param f The factor space for the internal PartialFactors.
                 * @param factors The factors to take into consideration.
                 */
                PartialFactorsEnumerator(Factors f, const std::vector<size_t> factors);

                /**
                 * @brief Skip constructor.
                 *
                 * This constructor is the same as the basic one, but it
                 * additionally remembers that the input factorToSkip will not
                 * be enumerated, and will in fact be editable by the client.
                 *
                 * The factorToSkip must be contained in the factors, or it
                 * will not be taken into consideration.
                 *
                 * @param f The factor space for the internal PartialFactors.
                 * @param factors The factors to take into considerations.
                 * @param factorToSkip The factor to skip.
                 */
                PartialFactorsEnumerator(Factors f, const std::vector<size_t> factors, size_t factorToSkip);

                /**
                 * @brief This function returns the id of the factorToSkip inside the PartialFactors.
                 *
                 * This function is provided for convenience, since
                 * PartialFactorsEnumerator has to compute this id anyway. It
                 * represents the id of the factorToSkip inside the vectors
                 * contained in the PartialFactors. This is useful so the
                 * client can go and edit that particular element directly.
                 *
                 * @return The id of the factorToSkip inside the PartialFactors.
                 */
                size_t getFactorToSkipId() const;

                /**
                 * @brief This function advances the PartialFactors to the next possible combination.
                 */
                void advance();

                /**
                 * @brief This function returns whether this object has terminated advancing and can be dereferenced.
                 *
                 * @return True if we can still be dereferenced, false otherwise.
                 */
                bool isValid() const;

                /**
                 * @brief This operator returns the current iteration in the values of the PartialFactors.
                 *
                 * This operator can be called only if isValid() is true.
                 * Otherwise behavior is undefined.
                 *
                 * The PartialFactors returned are editable, so that the user
                 * can change the factorToSkip values. If other values are
                 * edited, and the resulting PartialFactors still has valid
                 * values (below the Factors ceiling), the next advance() call
                 * will continue from there. If advance() is called with an
                 * invalid PartialFactors behavior is undefined.
                 *
                 * @return The current PartialFactors values.
                 */
                PartialFactors& operator*();

            private:
                Factors F;
                PartialFactors factors_;
                size_t factorToSkipId_;
        };
    }
}

#endif
