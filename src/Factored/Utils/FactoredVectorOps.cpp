#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    BasisFunction dot(const Factors & space, const BasisFunction & lhs, const BasisFunction & rhs) {
        BasisFunction retval;

        // The output function will have the domain of both inputs.
        retval.tag = merge(lhs.tag, rhs.tag);

        retval.values.resize(toIndexPartial(retval.tag, space, space));
        // No need to zero fill

        PartialFactorsEnumerator e(space, retval.tag);
        for (size_t i = 0; e.isValid(); e.advance(), ++i) {
            // We don't need to compute the index for retval since it
            // increases sequentially anyway.
            const auto lhsId = toIndexPartial(lhs.tag, space, *e);
            const auto rhsId = toIndexPartial(rhs.tag, space, *e);

            retval.values[i] = lhs.values[lhsId] * rhs.values[rhsId];
        }
        return retval;
    }

    BasisFunction plus(const Factors & space, const BasisFunction & lhs, const BasisFunction & rhs) {
        BasisFunction retval;

        // The output function will have the domain of both inputs.
        retval.tag = merge(lhs.tag, rhs.tag);

        retval.values.resize(toIndexPartial(retval.tag, space, space));
        // No need to zero fill

        PartialFactorsEnumerator e(space, retval.tag);
        for (size_t i = 0; e.isValid(); e.advance(), ++i) {
            // We don't need to compute the index for retval since it
            // increases sequentially anyway.
            const auto lhsId = toIndexPartial(lhs.tag, space, *e);
            const auto rhsId = toIndexPartial(rhs.tag, space, *e);

            retval.values[i] = lhs.values[lhsId] + rhs.values[rhsId];
        }
        return retval;
    }

    BasisFunction minus(const Factors & space, const BasisFunction & lhs, const BasisFunction & rhs) {
        BasisFunction retval;

        // The output function will have the domain of both inputs.
        retval.tag = merge(lhs.tag, rhs.tag);

        retval.values.resize(toIndexPartial(retval.tag, space, space));
        // No need to zero fill

        PartialFactorsEnumerator e(space, retval.tag);
        for (size_t i = 0; e.isValid(); e.advance(), ++i) {
            // We don't need to compute the index for retval since it
            // increases sequentially anyway.
            const auto lhsId = toIndexPartial(lhs.tag, space, *e);
            const auto rhsId = toIndexPartial(rhs.tag, space, *e);

            retval.values[i] = lhs.values[lhsId] - rhs.values[rhsId];
        }
        return retval;
    }

    BasisFunction & plusEqualSubset(const Factors & space, BasisFunction & retval, const BasisFunction & rhs) {
        if (retval.tag.size() == rhs.tag.size()) {
            retval.values += rhs.values;
            return retval;
        }
        PartialFactorsEnumerator e(space, retval.tag);
        for (size_t i = 0; e.isValid(); e.advance(), ++i) {
            const auto rhsId = toIndexPartial(rhs.tag, space, *e);

            retval.values[i] += rhs.values[rhsId];
        }
        return retval;
    }

    BasisFunction plusSubset(const Factors & space, BasisFunction retval, const BasisFunction & rhs) {
        return plusEqualSubset(space, retval, rhs);
    }

    BasisFunction & minusEqualSubset(const Factors & space, BasisFunction & retval, const BasisFunction & rhs) {
        if (retval.tag.size() == rhs.tag.size()) {
            retval.values -= rhs.values;
            return retval;
        }
        PartialFactorsEnumerator e(space, retval.tag);
        for (size_t i = 0; e.isValid(); e.advance(), ++i) {
            const auto rhsId = toIndexPartial(rhs.tag, space, *e);

            retval.values[i] -= rhs.values[rhsId];
        }
        return retval;
    }

    BasisFunction minusSubset(const Factors & space, BasisFunction retval, const BasisFunction & rhs) {
        return minusEqualSubset(space, retval, rhs);
    }

    FactoredVector & plusEqual(const Factors & space, FactoredVector & retval, const BasisFunction & basis) {
        const size_t initRetSize = retval.bases.size();

        // We try to merge all possible
        bool merged = false;
        for (size_t i = 0; i < initRetSize; ++i) {
            auto & curBasis = retval.bases[i];

            const auto retvalBigger = basis.tag.size() <= curBasis.tag.size();
            const auto & minBasis = retvalBigger ? basis : curBasis;
            const auto & maxBasis = retvalBigger ? curBasis : basis;

            if (sequential_sorted_contains(maxBasis.tag, minBasis.tag)) {
                if (retvalBigger)
                    plusEqualSubset(space, curBasis, basis);
                else
                    curBasis = plusSubset(space, basis, curBasis);
                merged = true;
                break;
            }
        }
        if (!merged)
            retval.bases.push_back(basis);

        return retval;
    }

    FactoredVector & plusEqual(const Factors & space, FactoredVector & retval, BasisFunction && basis) {
        const size_t initRetSize = retval.bases.size();

        // We try to merge all possible
        bool merged = false;
        for (size_t i = 0; i < initRetSize; ++i) {
            auto & curBasis = retval.bases[i];

            const auto retvalBigger = basis.tag.size() <= curBasis.tag.size();
            const auto & minBasis = retvalBigger ? basis : curBasis;
            const auto & maxBasis = retvalBigger ? curBasis : basis;

            if (sequential_sorted_contains(maxBasis.tag, minBasis.tag)) {
                if (retvalBigger)
                    plusEqualSubset(space, curBasis, basis);
                else
                    curBasis = std::move(plusEqualSubset(space, basis, curBasis));
                merged = true;
                break;
            }
        }
        if (!merged)
            retval.bases.emplace_back(std::move(basis));

        return retval;
    }

    FactoredVector plus(const Factors & space, FactoredVector retval, const BasisFunction & rhs) {
        return plusEqual(space, retval, rhs);
    }

    FactoredVector & minusEqual(const Factors & space, FactoredVector & retval, const BasisFunction & basis, bool clearZero) {
        const size_t initRetSize = retval.bases.size();

        // We try to merge all possible
        bool merged = false;
        for (size_t i = 0; i < initRetSize; ++i) {
            auto & curBasis = retval.bases[i];

            const auto retvalBigger = basis.tag.size() <= curBasis.tag.size();
            const auto & minBasis = retvalBigger ? basis : curBasis;
            const auto & maxBasis = retvalBigger ? curBasis : basis;

            if (sequential_sorted_contains(maxBasis.tag, minBasis.tag)) {
                if (retvalBigger)
                    plusEqualSubset(space, curBasis, basis);
                else
                    curBasis = plusSubset(space, basis, curBasis);
                merged = true;

                // If the basis is now useless, we remove it.
                if (clearZero && checkEqualGeneral(curBasis.values, 0.0))
                    retval.bases.erase(std::begin(retval.bases) + i);

                break;
            }
        }
        if (!merged)
            retval.bases.push_back(basis);

        return retval;
    }

    FactoredVector plus(const Factors & space, FactoredVector retval, const FactoredVector & rhs) {
        return plusEqual(space, retval, rhs);
    }

    FactoredVector & plusEqual(const Factors & space, FactoredVector & retval, const FactoredVector & rhs) {
        for (const auto & basis : rhs.bases)
            plusEqual(space, retval, basis);

        return retval;
    }

    FactoredVector & plusEqual(const Factors & space, FactoredVector & retval, FactoredVector && rhs) {
        for (auto && basis : rhs.bases)
            plusEqual(space, retval, std::move(basis));

        return retval;
    }

    FactoredVector minus(const Factors & space, FactoredVector retval, const BasisFunction & rhs, bool clearZero) {
        return minusEqual(space, retval, rhs, clearZero);
    }

    FactoredVector minus(const Factors & space, FactoredVector retval, const FactoredVector & rhs, bool clearZero) {
        return minusEqual(space, retval, rhs, clearZero);
    }

    FactoredVector & minusEqual(const Factors & space, FactoredVector & retval, const FactoredVector & rhs, bool clearZero) {
        for (const auto & basis : rhs.bases)
            minusEqual(space, retval, basis, clearZero);

        return retval;
    }

}
