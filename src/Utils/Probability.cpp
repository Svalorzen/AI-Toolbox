#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox {
    ProbabilityVector projectToProbability(const Vector & v) {
        ProbabilityVector retval(v.size());

        double sum = 0.0;
        size_t count = 0;
        for (auto i = 0; i < v.size(); ++i) {
            // Negative elements are converted to zero, as that's the best we
            // can do.
            if (v[i] < 0.0) retval[i] = 0.0;
            else {
                retval[i] = 1.0;
                ++count;
                sum += v[i];
            }
        }
        if (checkEqualSmall(sum, 1.0)) return retval;
        if (checkEqualSmall(sum, 0.0)) {
            // Any solution here would do, but this seems nice.
            retval.array() += 1.0 / v.size();
        } else if (sum > 1.0) {
            // We normalize the vector.
            retval.array() *= v.array() / sum;
        } else {
            // We remove equally from all non-zero elements.
            const auto diff = (1.0 - sum) / count;
            retval.array() *= (v.array() + diff);
        }
        return retval;
    }

    VoseAliasSampler::VoseAliasSampler(const ProbabilityVector & p) :
            prob_(p), alias_(prob_.size()), sampleDistribution_(0, prob_.size())
    {
        // Here we do the Vose Alias setup in a way that avoids the creation of
        // the small and large arrays.
        //
        // In practice what we do is we keep two pointers, one for large
        // elements and one for the small ones, and we move them along the
        // array as if we had already sorted the thing.

        const auto avg = 1.0 / prob_.size();
        auto small = 0, large = 0;
        while (small < prob_.size() && prob_[small] >= avg) ++small;
        while (large < prob_.size() && prob_[large] < avg) ++large;

        auto smallCheckpoint = small;

        while (small < prob_.size() && large < prob_.size()) {
            // Note: we do not do any assignments to prob_[small] here since if
            // we scaled the values already we might trip the large counter (as
            // it might be behind the small counter).
            prob_[large] = (prob_[large] + prob_[small]) - avg;
            alias_[small] = large;

            // If the large became small, we temporarily move the small counter
            // here, and look around for a new large element.
            // Otherwise, we go back to our last small 'checkpoint', and we
            // look for a new small element.
            if (prob_[large] < avg) {
                small = large;
                ++large;
                while (large < prob_.size() && prob_[large] < avg) ++large;
            } else {
                small = smallCheckpoint + 1;
                while (small < prob_.size() && prob_[small] >= avg) ++small;
                // Set the checkpoint again
                smallCheckpoint = small;
            }
        }

        // Now, for each entry which remained unassigned (so it is still with
        // the 0 default in the alias vector), we set it to just reference
        // itself. This takes care of both large and small entries which have
        // been left with no pairings.
        auto x = std::min(large, small);
        while (x < prob_.size()) {
            prob_[x] = 1.0;
            alias_[x] = x;
            ++x;
            while (x < prob_.size() && alias_[x] != 0) ++x;
        }

        // Here we scale up the vector so that each entry can be correctly seen
        // as a weighted coin. Note that all 1.0 entries will now be larger,
        // but for those there's no choice so we don't care about the precise
        // value anyway.
        prob_ *= prob_.size();
    }
}
