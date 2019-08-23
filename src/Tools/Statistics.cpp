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
        for (const auto & d : data_) {
            const auto & [count, sum, square, sqsum] = d;

            const double mean = sum / count;
            const double variance = square / count - mean * mean;
            const double std = std::sqrt(variance);
            cumMean += mean;
            const double cumVariance = sqsum / count - cumMean * cumMean;
            const double cumStd = std::sqrt(cumVariance);

            retval.emplace_back(mean, cumMean, std, cumStd);
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
