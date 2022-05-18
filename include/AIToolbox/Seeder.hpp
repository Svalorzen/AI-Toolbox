#ifndef AI_TOOLBOX_SEEDER_HEADER_FILE
#define AI_TOOLBOX_SEEDER_HEADER_FILE

#include <random>
#include <AIToolbox/Types.hpp>

namespace AIToolbox {
    /**
     * @brief This class is an internal class used to seed all random engines in the library.
     *
     * To avoid seeding all generators with a single seed equal to the current time, only
     * this class is setup with the time seed, while all others are seeded with numbers
     * generated from this class to obtain maximum randomness.
     */
    class Seeder {
        public:
            /**
             * @brief This function gets a random number to seed generators.
             *
             * @return A random unsigned number.
             */
            static unsigned getSeed();

            /**
             * @brief This function sets the seed for the seed generator.
             *
             * By default the generator is seeded with the current time. If
             * this is not satisfactory, due for example to the need of having
             * reproducible experiments, this function can be called in order
             * to seed the underlying generator.
             *
             * @param seed The seed for the underlying generator.
             */
            static void setRootSeed(unsigned seed);

            /**
             * @brief This function returns the root seed of Seeder.
             *
             * This works even if it has not been set manually. In this way it
             * is possible to log the original seed of a run to replicate it.
             *
             * @return The last set root seed.
             */
            static unsigned getRootSeed();

        private:
            Seeder();

            static Seeder instance_;

            unsigned rootSeed_;
            RandomEngine generator_;
    };
}

#endif
