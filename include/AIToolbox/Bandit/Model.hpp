#ifndef AI_TOOLBOX_BANDIT_MODEL_HEADER_FILE
#define AI_TOOLBOX_BANDIT_MODEL_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class represent a multi-armed bandit.
     *
     * This class contains a set of distributions, each of which corresponds to
     * a specific bandit arm. The arms are all assumed to be of the same
     * family; we could work with different distributions but it would
     * complicate the code for something that is not really commonly used.
     *
     * The class is fairly easy to use, as one can only pull a given arm and
     * obtain a sampled reward in return.
     *
     * The distribution is assumed to be one of the standard C++ distributions.
     * Custom ones may be used, as long as they can be sampled by passing a
     * RandomEngine to their operator().
     *
     * @tparam Dist The distribution family to use for all arms.
     */
    template <typename Dist>
    class Model {
        public:
            /**
             * @brief Basic constructor.
             *
             * We take as input a variable number of tuples (possibly
             * containing different types). Each tuple is used to initialize a
             * single arm.
             *
             * The number of arms will be equal to the number of tuples passed
             * as arguments.
             *
             * @param tupleArgs A set tuples, each containing the parameters to initialize an arm.
             */
            template <typename... TupleArgs>
            Model(TupleArgs... tupleArgs);

            /**
             * @brief Basic constructor.
             *
             * This constructor initializes each arm from one of the tuples
             * contained by the parameter.
             *
             * The number of arms will be equal to the size of the input
             * vector.
             *
             * @param args The arguments with which to initialize the bandit arms.
             */
            template <typename... Args>
            Model(std::vector<std::tuple<Args...>> args);

            /**
             * @brief This function samples the specified bandit arm.
             *
             * @param a The arm to sample.
             *
             * @return A return sampled from the arm's underlying distribution.
             */
            double sampleR(size_t a) const;

            /**
             * @brief This function returns the number of arms of the bandit.
             *
             * @return The number of arms of the bandit.
             */
            size_t getA() const;

            /**
             * @brief This function returns a reference to the underlying arms.
             *
             * @return A vector containing the arms of the bandit.
             */
            const std::vector<Dist> & getArms() const;

        private:
            mutable std::vector<Dist> arms_;
            mutable AIToolbox::RandomEngine rand_;
    };

    template <typename Dist>
    template <typename... TupleArgs>
    Model<Dist>::Model(TupleArgs... tupleArgs) :
        arms_({std::make_from_tuple<Dist>(std::move(tupleArgs))...}), rand_(AIToolbox::Impl::Seeder::getSeed())
    {}

    template <typename Dist>
    template <typename... Args>
    Model<Dist>::Model(std::vector<std::tuple<Args...>> args) :
        rand_(AIToolbox::Impl::Seeder::getSeed())
    {
        arms_.reserve(args.size());

        // Here we use a lambda to avoid having to static_cast the correct
        // emplace_back method on the vector.
        for (auto && t : args)
            std::apply([this](auto&&... params){arms_.emplace_back(std::move(params)...);}, std::move(t));
    }

    template <typename Dist>
    double Model<Dist>::sampleR(const size_t a) const {
        return arms_[a](rand_);
    }

    template <typename Dist>
    size_t Model<Dist>::getA() const { return arms_.size(); }

    template <typename Dist>
    const std::vector<Dist> & Model<Dist>::getArms() const { return arms_; }
}

#endif
