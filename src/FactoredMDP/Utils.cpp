#include <AIToolbox/FactoredMDP/Utils.hpp>

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        PartialFactors removeFactor(const PartialFactors & pf, size_t f) {
            size_t i = 0;
            while (i < pf.first.size() && pf.first[i] < f) ++i;
            if (i == pf.first.size() || pf.first[i] != f) return pf;

            PartialFactors retval; 
            retval.first.reserve(pf.first.size() - 1);
            retval.second.reserve(pf.first.size() - 1);

            for (size_t j = 0; j < pf.first.size(); ++j) {
                if (i == j) continue;
                retval.first.push_back(pf.first[j]);
                retval.second.push_back(pf.second[j]);
            }
            return retval;
        }
        bool match(const PartialFactors & lhs, const PartialFactors & rhs) {
            const PartialFactors * smaller = &rhs, * bigger = &lhs;
            if (lhs.first.size() < rhs.first.size()) std::swap(smaller, bigger);

            size_t i = 0, j = 0;
            while (j < smaller->second.size()) {
                if (bigger->first[i] < smaller->first[j]) ++i;
                else if (bigger->first[i] > smaller->first[j]) return false;
                else {
                    if (bigger->second[i] != smaller->second[j]) return false;
                    ++i;
                    ++j;
                }
            }
            return true;
        }

        PartialFactors join(size_t S, const PartialFactors & lhs, const PartialFactors & rhs) {
            PartialFactors retval;
            retval.first.reserve(lhs.first.size() + rhs.first.size());
            retval.second.reserve(lhs.first.size() + rhs.first.size());
            // lhs part is the same.
            retval = lhs;
            // rhs part is shifted by S elements (values are the same)
            std::transform(std::begin(rhs.first), std::end(rhs.first), std::back_inserter(retval.first), [S](size_t a){ return a + S; });
            retval.second.insert(std::end(retval.second), std::begin(rhs.second), std::end(rhs.second));

            return retval;
        }

        Factors join(const Factors & lhs, const Factors & rhs) {
            Factors retval;
            retval.reserve(lhs.size() + rhs.size());
            retval.insert(std::end(retval), std::begin(lhs), std::end(lhs));
            retval.insert(std::end(retval), std::begin(rhs), std::end(rhs));
            return retval;
        }

        PartialFactors merge(const PartialFactors & lhs, const PartialFactors & rhs) {
            PartialFactors retval;
            retval.first.reserve(lhs.first.size() + rhs.first.size());
            retval.second.reserve(lhs.first.size() + rhs.first.size());
            retval = lhs;

            inplace_merge(&retval, rhs);

            return retval;
        }

        void inplace_merge(PartialFactors * plhs, const PartialFactors & rhs) {
            if (!plhs) return;
            auto & lhs = *plhs;

            lhs.first.reserve(lhs.first.size() + rhs.first.size());
            lhs.second.reserve(lhs.first.size() + rhs.first.size());

            size_t i = 0, j = 0;
            while (i < lhs.first.size() && j < rhs.first.size()) {
                if (lhs.first[i] < rhs.first[j]) { ++i; continue; }
                lhs.first.insert(std::begin(lhs.first) + i, rhs.first[j]);
                lhs.second.insert(std::begin(lhs.second) + i, rhs.second[j]);
                ++i;
                ++j;
            }
            lhs.first.insert(std::end(lhs.first), std::begin(rhs.first) + j, std::end(rhs.first));
            lhs.second.insert(std::end(lhs.second), std::begin(rhs.second) + j, std::end(rhs.second));
        }


        size_t factorSpace(const Factors & factors) {
            size_t retval = 1;
            for (auto f : factors) {
                // Detect wraparound
                if (std::numeric_limits<size_t>::max() / f < retval)
                    return std::numeric_limits<size_t>::max();
                retval *= f;
            }
            return retval;
        }

        PartialFactors toPartialFactors(const Factors & f) {
            PartialFactors retval;

            retval.first.resize(f.size());
            for (size_t i = 0; i < f.size(); ++i)
                retval.first[i] = i;
            retval.second = f;

            return retval;
        }

        Factors toFactors(size_t F, const PartialFactors & pf) {
            Factors f(F);
            for (size_t i = 0; i < pf.first.size(); ++i)
                f[pf.first[i]] = pf.second[i];

            return f;
        }

        // PartialFactorsEnumerator below.

        PartialFactorsEnumerator::PartialFactorsEnumerator(Factors f, const std::vector<size_t> factors) :
            F(f), factorToSkipId_(factors.size())
        {
            factors_.first = factors;
            factors_.second.resize(factors.size());
        }

        PartialFactorsEnumerator::PartialFactorsEnumerator(Factors f, const std::vector<size_t> factors, size_t factorToSkip) :
            F(f), factorToSkipId_(factors.size())
        {
            factors_.first.reserve(factors.size());
            factors_.second.resize(factors.size());
            // Set all used agents and find the skip id.
            for (size_t i = 0; i < factors.size(); ++i) {
                factors_.first.push_back(factors[i]);
                if (factorToSkip == factors[i])
                    factorToSkipId_ = i;
            }
        }

        void PartialFactorsEnumerator::advance() {
            // Start from 0 if skip is not zero, from 1 otherwise.
            size_t id = !factorToSkipId_;
            while (id < factors_.second.size()) {
                ++factors_.second[id];
                if (factors_.second[id] == F[factors_.first[id]]) {
                    factors_.second[id] = 0;
                    if (++id == factorToSkipId_) ++id;
                } else
                    return;
            }
            factors_.second.clear();
        }

        bool PartialFactorsEnumerator::isValid() const {
            return factors_.second.size() > 0;
        }

        size_t PartialFactorsEnumerator::getFactorToSkipId() const { return factorToSkipId_; }

        PartialFactors& PartialFactorsEnumerator::operator*() { return factors_; }
    }
}
