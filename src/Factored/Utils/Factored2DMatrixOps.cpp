#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    BasisMatrix plus(const Factors & space, const Factors & actions, const BasisMatrix & lhs, const BasisMatrix & rhs);
    BasisMatrix plusSubset(const Factors & space, const Factors & actions, BasisMatrix retval, const BasisMatrix & rhs) {
        return plusEqualSubset(space, actions, retval, rhs);
    }

    BasisMatrix & plusEqualSubset(const Factors & space, const Factors & actions, BasisMatrix & retval, const BasisMatrix & rhs) {
        if (retval.tag.size() == rhs.tag.size() &&
            retval.actionTag.size() == rhs.actionTag.size())
        {
            retval.values += rhs.values;
            return retval;
        }

        size_t x = 0, y = 0;
        PartialFactorsEnumerator e(space, retval.tag);
        PartialFactorsEnumerator a(actions, retval.actionTag);
        while (e.isValid()) {
            const auto rX = toIndexPartial(rhs.tag, space, *e);

            while (a.isValid()) {
                const auto rY = toIndexPartial(rhs.tag, space, *e);
                retval.values(x, y) += rhs.values(rX, rY);

                ++y;
                a.advance();
            }
            y = 0;
            a.reset();

            ++x;
            e.advance();
        }
        return retval;
    }

    Factored2DMatrix & plusEqual(const Factors & space, const Factors & actions, Factored2DMatrix & retval, const BasisMatrix & basis) {
        const size_t initRetSize = retval.bases.size();

        // We try to merge all possible
        bool merged = false;
        for (size_t i = 0; i < initRetSize; ++i) {
            auto & curBasis = retval.bases[i];

            const auto retvalBigger = basis.tag.size() <= curBasis.tag.size();
            const auto & minBasis = retvalBigger ? basis : curBasis;
            const auto & maxBasis = retvalBigger ? curBasis : basis;

            // We can only try to merge if the bigger one is bigger for both
            // tag and actionTag. If it's not, merging is not necessarily a
            // good trade so we just avoid it.
            if (maxBasis.actionTag.size() >= minBasis.actionTag.size() &&
                sequential_sorted_contains(maxBasis.actionTag, minBasis.actionTag) &&
                sequential_sorted_contains(maxBasis.tag, minBasis.tag))
            {
                if (retvalBigger)
                    plusEqualSubset(space, actions, curBasis, basis);
                else
                    curBasis = plusSubset(space, actions, basis, curBasis);
                merged = true;
                break;
            }
        }
        if (!merged)
            retval.bases.push_back(basis);

        return retval;
    }

    Factored2DMatrix & plusEqual(const Factors & space, const Factors & actions, Factored2DMatrix & retval, BasisMatrix && basis) {
        const size_t initRetSize = retval.bases.size();

        // We try to merge all possible
        bool merged = false;
        for (size_t i = 0; i < initRetSize; ++i) {
            auto & curBasis = retval.bases[i];

            const auto retvalBigger = basis.tag.size() <= curBasis.tag.size();
            const auto & minBasis = retvalBigger ? basis : curBasis;
            const auto & maxBasis = retvalBigger ? curBasis : basis;

            // We can only try to merge if the bigger one is bigger for both
            // tag and actionTag. If it's not, merging is not necessarily a
            // good trade so we just avoid it.
            if (maxBasis.actionTag.size() >= minBasis.actionTag.size() &&
                sequential_sorted_contains(maxBasis.actionTag, minBasis.actionTag) &&
                sequential_sorted_contains(maxBasis.tag, minBasis.tag))
            {
                if (retvalBigger)
                    plusEqualSubset(space, actions, curBasis, basis);
                else
                    curBasis = std::move(plusEqualSubset(space, actions, basis, curBasis));
                merged = true;
                break;
            }
        }
        if (!merged)
            retval.bases.emplace_back(std::move(basis));

        return retval;
    }

    Factored2DMatrix & plusEqual(const Factors & space, const Factors & actions, Factored2DMatrix & retval, const Factored2DMatrix & rhs) {
        for (const auto & basis : rhs.bases)
            plusEqual(space, actions, retval, basis);
        return retval;
    }

    Factored2DMatrix & plusEqual(const Factors & space, const Factors & actions, Factored2DMatrix & retval, Factored2DMatrix && rhs) {
        for (auto && basis : rhs.bases)
            plusEqual(space, actions, retval, std::move(basis));
        return retval;
    }
}
