#include <AIToolbox/Tools/Statistics.hpp>

#include <iostream>
#include <tuple>
#include <cmath>

namespace AIToolbox {
    Statistics::Statistics(size_t timesteps) :
            data_(timesteps), prevTimestep_(0), currentCumulativeValue_(0.0) {}

    void Statistics::record(const double v, const size_t t) {
        auto & [count, sum, square, sqsum] = data_[t];

        // Reset current cumulative value for this experiment in case we are in
        // a new run.
        if (t <= prevTimestep_) currentCumulativeValue_ = 0.0;
        prevTimestep_ = t;

        currentCumulativeValue_ += v;

        ++count;
        sum += v;
        square += v * v;
        sqsum += currentCumulativeValue_ * currentCumulativeValue_;
    }

    Statistics::Results Statistics::process() const {
        Results retval;
        retval.reserve(data_.size());

        double cumMean = 0.0;
        double cumSum = 0.0;
        for (const auto & d : data_) {
            const auto & [count, sum, square, sqsum] = d;

            const double mean = sum / count;
            // The max is to avoid floating point negatives which create nan with sqrt.
            //
            // Note that the std will be biased since we do a non-linear
            // transform on the unbiased variance. The fix for this however
            // depend on the distribution we're tracking, so we cna't do much
            // here.
            const double unbiasedVariance = std::max(0.0, (square - mean * sum) / (count - 1.0));
            const double biasedStd = std::sqrt(unbiasedVariance);

            cumSum += sum;
            cumMean += mean;

            // Same as above.
            const double cumUnbiasedVariance = std::max(0.0, (sqsum - cumMean * cumSum) / (count - 1.0));
            const double cumBiasedStd = std::sqrt(cumUnbiasedVariance);

            retval.emplace_back(mean, cumMean, biasedStd, cumBiasedStd);
        }
        return retval;
    }

    std::ostream& operator<<(std::ostream& os, const Statistics & rh) {
        const auto data = rh.process();

        for (unsigned t = 0; t < data.size(); ++t) {
            const auto & [mean, cumMean, std, cumStd] = data[t];
            os << t << ' ' << mean << ' ' << cumMean << ' ' << std << ' ' << cumStd << '\n';
        }

        return os;
    }
}
