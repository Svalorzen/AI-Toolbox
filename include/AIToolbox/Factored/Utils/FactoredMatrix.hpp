#ifndef AI_TOOLBOX_FACTORED_UTILS_FACTORED_MATRIX_HEADER_FILE
#define AI_TOOLBOX_FACTORED_UTILS_FACTORED_MATRIX_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>

namespace AIToolbox::Factored {
    /**
     * @name Factored Matrix Types
     *
     * Here we define some factored matrix types. These types allow to specify
     * matrices in a factored form, i.e. where a matrix can be seen as a sum or
     * product of smaller matrices, defined on subsets of the state-space.
     *
     * @{
     */

    /**
     * @brief This struct represents a basis function.
     *
     * A basis function is simply a function defined on some subset of factors.
     * It maps each combination of values that can be taken by the factors with
     * a double.
     *
     * Note that, as far as the math is concerned, a basis function is a
     * function over the whole factor space anyway. It's simply zero where it
     * is not defined.
     *
     * Thus, values will always have a number of elements equal to
     * factorSpacePartial(tag, space), where space is the factor space this
     * basis applies to.
     */
    struct BasisFunction {
        PartialKeys tag;
        Vector values;
    };

    /**
     * @brief This class represents a factored vector.
     *
     * A factored vector is a function over the whole factor space, resulting
     * from the sum of a set of BasisFunctions. This representation turns out
     * to be convenient as it allows to do many operations much more
     * efficiently than if we had to represent the function explicitly.
     *
     * This representation is used, for example, to represent each Qa, or Ra in
     * factored MDPs.
     *
     * Note that, in most cases, one should try to avoid duplicate
     * BasisFunctions in here; i.e.  BasisFunctions with equal tag, or where
     * the tag of one is a strict superset of another. Such BasisFunctions can
     * simply be merged, thus keeping the representation as simple as possible.
     *
     * At the same time, sometimes duplicates are needed: for example when
     * specifying the BasisFunctions of a factored ValueFunction. This is
     * because usually a factored ValueFunction is composed of two components:
     * the BasisFunctions, and a set of weight. Being able to assign separate
     * weights to two different BasisFunctions, even if defined over the same
     * tag, is important, so we don't completely forbid duplicates.
     */
    struct FactoredVector {
        /**
         * @brief This function returns the value of the FactoredVector at the specified point.
         *
         * @param space The factor space to use.
         * @param value The value of the factor space to compute the value for.
         *
         * @return The value of the FactoredVector at the specified point.
         */
        double getValue(const Factors & space, const Factors & value) const;

        /**
         * @brief This function multiplies each basis function with a weight.
         *
         * The input Vector *must* have the same number of elements as the
         * number of bases in this FactoredVector.
         *
         * Each basis is multiplied by its respective scalar.
         *
         * @param w The weights to multiply with.
         *
         * @return A reference to this FactoredVector.
         */
        FactoredVector & operator*=(const Vector & w);

        /**
         * @brief This function multiplies all bases with a scalar.
         *
         * @param v The scalar to multiply with.
         *
         * @return A reference to this FactoredVector.
         */
        FactoredVector & operator*=(const double v);

        std::vector<BasisFunction> bases;
    };

    /**
     * @brief This struct represents a basis matrix.
     *
     * Note that the term "basis matrix" does not really exist in the
     * literature, it's just a way to create a basis which depends on both
     * state factors and action factors at the same time.
     *
     * A basis matrix here is simply a function defined on two subsets of
     * factors. It maps each combination of values that can be taken by the
     * factors with a double.
     *
     * The matrix is SxA; where S and A are the factorSpacePartial() of the two
     * tags.
     */
    struct BasisMatrix {
        PartialKeys tag;
        PartialKeys actionTag;
        Matrix2D values;
    };

    /**
     * @brief This class represents a factored 2D matrix.
     *
     * Note that we can't use a multi_array since each FactoredVector may have
     * a different number of BasisFunctions.
     */
    struct Factored2DMatrix {
        /**
         * @brief This function returns the value of the FactoredVector at the specified point.
         *
         * @param space The factor space to use.
         * @param actions The action space to use.
         * @param value The factor to compute the value for.
         * @param action The action to compute the value for.
         *
         * @return The value of the FactoredVector at the specified point.
         */
        double getValue(const Factors & space, const Factors & actions, const Factors & value, const Factors & action) const;

        /**
         * @brief This function multiplies all bases with a scalar.
         *
         * @param v The scalar to multiply with.
         *
         * @return A reference to this FactoredVector.
         */
        Factored2DMatrix & operator*=(const double v);

        std::vector<BasisMatrix> bases;
    };

    // @}

    // BasisFunction - BasisFunction

    BasisFunction dot(const Factors & space, const BasisFunction & lhs, const BasisFunction & rhs);
    BasisFunction plus(const Factors & space, const BasisFunction & lhs, const BasisFunction & rhs);
    BasisFunction minus(const Factors & space, const BasisFunction & lhs, const BasisFunction & rhs);

    BasisFunction plusSubset(const Factors & space, BasisFunction retval, const BasisFunction & rhs);
    BasisFunction minusSubset(const Factors & space, BasisFunction retval, const BasisFunction & rhs);

    BasisFunction & plusEqualSubset(const Factors & space, BasisFunction & retval, const BasisFunction & rhs);
    BasisFunction & minusEqualSubset(const Factors & space, BasisFunction & retval, const BasisFunction & rhs);

    // BasisMatrix - BasisMatrix

    BasisMatrix plus(const Factors & space, const BasisMatrix & lhs, const BasisMatrix & rhs);

    // FactoredVector - BasisFunction

    FactoredVector plus(const Factors & space, FactoredVector retval, const BasisFunction & rhs);
    FactoredVector & plusEqual(const Factors & space, FactoredVector & retval, const BasisFunction & basis);
    FactoredVector & plusEqual(const Factors & space, FactoredVector & retval, BasisFunction && basis);

    FactoredVector minus(const Factors & space, FactoredVector retval, const BasisFunction & rhs, bool clearZero = false);
    FactoredVector & minusEqual(const Factors & space, FactoredVector & retval, const BasisFunction & basis, bool clearZero = false);

    // Factored2DMatrix - BasisMatrix

    // TODO
    Factored2DMatrix & plusEqual(const Factors & space, const Factors & actions, Factored2DMatrix & retval, const BasisMatrix & basis);
    Factored2DMatrix & plusEqual(const Factors & space, const Factors & actions, Factored2DMatrix & retval, BasisMatrix && basis);

    // FactoredVector - FactoredVector

    FactoredVector plus(const Factors & space, FactoredVector retval, const FactoredVector & rhs);
    FactoredVector & plusEqual(const Factors & space, FactoredVector & retval, const FactoredVector & rhs);

    FactoredVector minus(const Factors & space, FactoredVector retval, const FactoredVector & rhs, bool clearZero = false);
    FactoredVector & minusEqual(const Factors & space, FactoredVector & retval, const FactoredVector & rhs, bool clearZero = false);

    // Factored2DMatrix - Factored2DMatrix

    // TODO
    Factored2DMatrix & plusEqual(const Factors & space, const Factors & actions, Factored2DMatrix & retval, const Factored2DMatrix & basis);
    Factored2DMatrix & plusEqual(const Factors & space, const Factors & actions, Factored2DMatrix & retval, Factored2DMatrix && basis);

    // Scalar ops

    FactoredVector operator*(FactoredVector lhs, const Vector & w);
    FactoredVector operator*(const Vector & w, FactoredVector rhs);
    FactoredVector operator*(FactoredVector lhs, const double v);
    FactoredVector operator*(const double v, FactoredVector rhs);

    Factored2DMatrix operator*(Factored2DMatrix lhs, const double v);
    Factored2DMatrix operator*(const double v, Factored2DMatrix rhs);
}

#endif
