#ifndef AI_TOOLBOX_FACTORED_UTILS_HEADER_FILE
#define AI_TOOLBOX_FACTORED_UTILS_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>

namespace AIToolbox::Factored {
    /**
     * @brief This function removes the specified factor from the input PartialFactors.
     *
     * @param pf The PartialFactors to modify.
     * @param f The factor to be removed.
     *
     * @return A new PartialFactors that does not contain the input factor.
     */
    PartialFactors removeFactor(const PartialFactors & pf, size_t f);

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
     * @brief This function returns whether the common factors in the inputs match in value.
     *
     * This function is equivalent to match(const PartialFactors & lhs, const
     * PartialFactors & rhs). It is provided to avoid having to construct
     * two PartialFactors when not needed.
     *
     * @param lhsK The keys of the lhs.
     * @param lhs The values of the lhs.
     * @param rhsK The keys of the rhs.
     * @param rhs The values of the rhs.
     *
     * @return True if all factors in common between the inputs match in value, false otherwise.
     */
    bool match(const PartialKeys & lhsK, const PartialValues & lhs, const PartialKeys & rhsK, const PartialValues & rhs);

    /**
     * @brief This function checks whether the two input Factors match at the specified ids.
     *
     * @param keys The ids to check.
     * @param lhs The left hand side.
     * @param rhs The right hand side.
     *
     * @return Whether the two Factors match at the specified ids.
     */
    bool match(const PartialKeys & keys, const Factors & lhs, const Factors & rhs);

    /**
     * @brief This function checks whether the two input Factors match at the specified ids.
     *
     * Each check is performed on a pair of ids: one for the left hand side,
     * and its respective one for the right hand side.
     *
     * \sa merge(const PartialKeys &, const PartialKeys &, std::vector<std::pair<size_t, size_t>> *)
     *
     * @param matches The id pairs to check.
     * @param lhs The left hand side.
     * @param rhs The right hand side.
     *
     * @return Whether the two Factors match at the specified ids.
     */
    bool match(const std::vector<std::pair<size_t, size_t>> & matches, const Factors & lhs, const Factors & rhs);

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
     * @brief This function appends the rhs to the lhs.
     *
     * This function may produce a non-valid PartialFactors. This is useful in
     * case multiple joins must be done in successions, so that the process is
     * more efficient.
     *
     * Remember to call sort() at the end to make the output valid again.
     *
     * @param lhs The left hand side that gets extended in place.
     * @param rhs The right hand side.
     */
    void unsafe_join(PartialFactors * lhs, const PartialFactors & rhs);

    /**
     * @brief This function merges two PartialFactors together.
     *
     * This function assumes that all elements in the PartialFactors have
     * different keys. If they have the same keys, the key is inserted once in
     * the output, but its value is unspecified (it will be from one of the two
     * inputs).
     *
     * @param lhs The left hand side.
     * @param rhs The right hand side.
     *
     * @return A new PartialFactors containing all keys from both inputs and their respective values.
     */
    PartialFactors merge(const PartialFactors & lhs, const PartialFactors & rhs);

    /**
     * @brief This function merges two Factors together, using two PartialKeys as guides.
     *
     * This function is equivalent to merge(const PartialFactors&, const PartialFactors&).
     *
     * It merges two Factors using their PartialKeys as guides. The Factors
     * must be of the same size as their respective PartialKeys.
     *
     * This function assumes that all elements in the PartialKeys are
     * different. If there are matches, the corrisponding element is inserted
     * once output, but its value is unspecified (it will be from one of the
     * two input Factors).
     *
     * @param lhsk The left hand side keys.
     * @param lhs The left hand side.
     * @param rhsk The right hand side keys.
     * @param rhs The right hand side.
     *
     * @return A new Factors containing all values from both inputs in the correct order.
     */
    Factors merge(const PartialKeys & lhsk, const Factors & lhs, const PartialKeys & rhsk, const Factors & rhs);

    /**
     * @brief This function merges two PartialKeys together.
     *
     * This function merges two PartialKeys over the same range. Overlapping
     * elements are merged together.
     *
     * The function optionally returns a vector which specifies the indeces of
     * the matches in the input. This may be useful to do checks before doing
     * merges.
     *
     * @param lhs The left hand side.
     * @param rhs The right hand side.
     * @param matches An optional vector containing the matching indexes of the inputs
     *
     * @return A new PartialKeys containing all unique keys from the inputs.
     */
    PartialKeys merge(const PartialKeys & lhs, const PartialKeys & rhs, std::vector<std::pair<size_t, size_t>> * matches = nullptr);

    /**
     * @brief This function returns the multiplication of all elements of the input factor.
     *
     * In case the factor space is too big to represent via a size_t, the
     * maximum possible representable value is returned.
     *
     * @param space The factored factor space.
     *
     * @return The possible number of factors if representable, otherwise the max size_t.
     */
    size_t factorSpace(const Factors & space);

    /**
     * @brief This function returns the multiplication of all elements of the input factor.
     *
     * This function only takes into account the input ids.
     *
     * In case the factor space is too big to represent via a size_t, the
     * maximum possible representable value is returned.
     *
     * \sa factorSpace
     *
     * @param space The factored factor space.
     *
     * @return The possible number of factors if representable, otherwise the max size_t.
     */
    size_t factorSpacePartial(const PartialKeys & ids, const Factors & space);

    /**
     * @brief This function converts Factors into the equivalent PartialFactors structure.
     *
     * @param f The Factors to be converted.
     *
     * @return A PartialFactors structure equivalent to the input.
     */
    PartialFactors toPartialFactors(const Factors & f);

    /**
     * @brief This function converts PartialFactors into the equivalent Factors structure.
     *
     * If the PartialFactors are incomplete, the non-specified elements
     * will be unspecified in the returned value as well. The
     * PartialFactors will not contain a factor higher than what passed as
     * input.
     *
     * @param F The size of the Factors to be returned.
     * @param pf The PartialFactors to be converted.
     *
     * @return Factors containing all values of the input.
     */
    Factors toFactors(size_t F, const PartialFactors & pf);

    /**
     * @brief This function converts an index into the equivalent Factors, within the specified factor space.
     *
     * This function is the inverse of the toIndex(const Factors &, size_t)
     * function.
     *
     * The input id shall not cause the output to exceed the input space
     * (i.e. the id will always be lower than factorSpace(space)).
     *
     * @param space The factor space to consider.
     * @param id The integer uniquely identifying the factor.
     *
     * @return The id's equivalent Factors.
     */
    Factors toFactors(const Factors & space, size_t id);

    /**
     * @brief This function converts an index into the equivalent Factors of the input keys, within the specified factor space.
     *
     * This function is the inverse of the toIndexPartial(const PartialKeys &,
     * const Factors &, const Factors &) function.
     *
     * The input id shall not cause the output to exceed the input space
     * (i.e. the id will always be lower than factorSpacePartial(ids, space)).
     *
     * @param ids The indeces to consider.
     * @param space The global factors space to consider.
     * @param id The integer uniquely identifying the factor.
     *
     * @return The id's equivalent Factors.
     */
    Factors toFactorsPartial(const PartialKeys & ids, const Factors & space, size_t id);

    /**
     * @brief This function converts the input factor in the input space to an unique index.
     *
     * This function returns an unique integer in range [0,
     * factorSpace(space)). The function makes **no guarantees** as to what
     * the conversion does, aside that from a unique factor it will return
     * always the same unique number.
     *
     * So for example in a space (2,3), a *possible* equivalency would be:
     *
     * (0,0) -> 0
     * (0,1) -> 1
     * (0,2) -> 2
     * (1,0) -> 3
     * (1,1) -> 4
     * (1,2) -> 5
     *
     * This function does not perform bound checking. If the resulting
     * number is too big to be representable via a size_t, the behavior is
     * unspecified. This can happen only if factorSpace(space) exceeds the
     * size of a size_t variable.
     *
     * @param space The factor space to consider.
     * @param f The input factor to convert.
     *
     * @return An integer which uniquely identifies the factor in the factor space.
     */
    size_t toIndex(const Factors & space, const Factors & f);

    /**
     * @brief This function converts the input factor in the input space to an unique index.
     *
     * \sa toIndex(const Factors &, const Factors &)
     *
     * All unspecified values are considered 0.
     *
     * @param space The factor space to consider.
     * @param f The input factor to convert.
     *
     * @return An integer which uniquely identifies the factor in the factor space.
     */
    size_t toIndex(const Factors & space, const PartialFactors & f);

    /**
     * @brief This function converts the input factor in the input space to an unique index.
     *
     * In this method ONLY the ids passed as input are considered. This
     * function effectively considers only a subset of the input space and
     * factor (which should still have the same length).
     *
     * So if the ids are {1, 3}, the function will behave as if the factor
     * space is two-dimensional, taking the values at ids 1 and 3 from the
     * input space.
     *
     * Then it will take the values at ids 1 and 3 from the input factor
     * and use them to compute the equivalent number.
     *
     * \sa toIndex(const Factors &, const Factors &)
     *
     * @param ids The ids to consider in the input space and factor.
     * @param space The factor space to consider.
     * @param f The input factor to convert.
     *
     * @return An integer which uniquely identifies the factor in the factor space for the specified ids.
     */
    size_t toIndexPartial(const PartialKeys & ids, const Factors & space, const Factors & f);

    /**
     * @brief This function converts the input factor in the input space to an unique index.
     *
     * In this method ONLY the ids passed as input are considered. This
     * function effectively considers only a subset of the input space and
     * partial factor. The partial factor MUST contain the ids passed as input!
     *
     * So if the ids are {1, 3}, the function will behave as if the factor
     * space is two-dimensional, taking the values at ids 1 and 3 from the
     * input space.
     *
     * Then it will take the values at keys 1 and 3 from the input partial
     * factor and use them to compute the equivalent number.
     *
     * \sa toIndex(const Factors &, const Factors &)
     *
     * @param ids The ids to consider in the input space and factor.
     * @param space The factor space to consider.
     * @param f The input factor to convert.
     *
     * @return An integer which uniquely identifies the factor in the factor space for the specified ids.
     */
    size_t toIndexPartial(const PartialKeys & ids, const Factors & space, const PartialFactors & pf);

    /**
     * @brief This function converts the input factor in the input space to an unique index.
     *
     * In this function only the ids mentioned in the PartialFactors are
     * considered to be part of the space.
     *
     * \sa toIndexPartial(const PartialKeys &, const Factors &, const Factors &);
     *
     * @param space The factor space to consider.
     * @param f The input factor to convert.
     *
     * @return An integer which uniquely identifies the factor in the factor space for the factor's ids.
     */
    size_t toIndexPartial(const Factors & space, const PartialFactors & f);

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
            PartialFactorsEnumerator(Factors f, const PartialKeys factors);

            /**
             * @brief Basic constructor.
             *
             * This constructor initializes the internal PartialFactors with
             * the factors obtained as inputs. This constructor can be used
             * when one wants to iterate over all factors.
             *
             * @param f The factor space for the internal PartialFactors.
             */
            PartialFactorsEnumerator(Factors f);

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
            PartialFactorsEnumerator(Factors f, const PartialKeys factors, size_t factorToSkip);

            /**
             * @brief Skip constructor.
             *
             * This constructor is the same as the basic one, but it
             * additionally remembers that the input factorToSkip will not
             * be enumerated, and will in fact be editable by the client.
             *
             * This constructor can be used to enumerate over all factors.
             *
             * The factorToSkip must be contained in the factors, or it
             * will not be taken into consideration.
             *
             * @param f The factor space for the internal PartialFactors.
             * @param factorToSkip The factor to skip.
             */
            PartialFactorsEnumerator(Factors f, size_t factorToSkip);

            /**
             * @brief This function returns the id of the factorToSkip inside the PartialFactorsEnumerator.
             *
             * This function is provided for convenience, since
             * PartialFactorsEnumerator has to compute this id anyway. It
             * represents the id of the factorToSkip inside the vectors
             * contained in the PartialFactors. This is useful so the
             * client can go and edit that particular element directly.
             *
             * @return The id of the factorToSkip inside the PartialFactorsEnumerator.
             */
            size_t getFactorToSkipId() const;

            /**
             * @brief This function advances the PartialFactorsEnumerator to the next possible combination.
             */
            void advance();

            /**
             * @brief This function returns whether this object has terminated advancing and can be dereferenced.
             *
             * @return True if we can still be dereferenced, false otherwise.
             */
            bool isValid() const;

            /**
             * @brief This function resets the enumerator to the valid beginning (a fully zero PartialFactor).
             */
            void reset();

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

#endif
