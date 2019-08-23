#ifndef AI_TOOLBOX_STATISTICS_HEADER_FILE
#define AI_TOOLBOX_STATISTICS_HEADER_FILE

#include <vector>
#include <iosfwd>

namespace AIToolbox {
    /**
     * @brief This class registers sets of data and computes statistics about it.
     *
     * This class is used to record multiple sets of data in order to create
     * mean and standard deviation statistics from them in an efficient manner.
     *
     * This can be used for example to easily compute statistics on reward/regret.
     *
     * This class only performs basic bookkeeping during the recording of the
     * data, as processing standard deviation requires all datapoints to be
     * available.
     *
     * For each timestep, this class only stores a summary of the number of
     * points recorded there, their sum and the sum of their products. This
     * class is not going to remember every single datapoint passed to it.
     *
     * This class must know in advance the number of timesteps to consider, in
     * order to pre-allocate the data vector for maximum performance.
     */
    class Statistics {
        public:
            //                        mean,  cum mean,  std,  cum std
            using Result = std::tuple<double, double, double, double>;
            using Results = std::vector<Result>;

            /**
             * @brief Basic constructor.
             *
             * @param timesteps The number of timesteps to process.
             */
            Statistics(size_t timesteps);

            /**
             * @brief This function records a new datapoint for the specified timestep.
             *
             * This function stores the input in a way that allows to obtain both
             * mean and standard deviation from the data later.
             *
             * This function assumes that records are passed in order, for each
             * run. If a new record has a timestep less than or equal to the
             * previously passed timestep, it's going to assume that the new
             * record refers to a new experiment run. This is important to
             * compute the cumulative std statistic correctly, but does not
             * otherwise affect the other ones.
             *
             * @param value The value to register.
             * @param timestep The timestep of the value.
             */
            void record(double value, size_t timestep);

            /**
             * @brief This function computes mean and standard deviation for all timesteps.
             *
             * @return The mean and standard deviation for the recorded data.
             */
            Results process() const;

        private:
            //                       Count,    sum,    sum squared, squared sum
            using Point = std::tuple<unsigned, double, double,      double>;

            std::vector<Point> data_;
            size_t prevTimestep_;  /// Last recorded timestep
            double currentCumulativeValue_; /// Cumulative trace for the current experiment run to record.
    };

    /**
     * @brief This function writes the output of the Statistics to the stream.
     *
     * The output will contain a series of lines, each formed by: timestep, mean,
     * cumulative mean, standard deviation and cumulative standard deviation.
     *
     * The output is GnuPlot friendly!
     *
     * Note that each reprint will recompute the statistics from scratch, as
     * they are not cached.
     *
     * @param os The stream to write to.
     * @param rh The Statistics to get data from.
     *
     * @return The input stream.
     */
    std::ostream& operator<<(std::ostream& os, const Statistics & rh);
}

#endif
