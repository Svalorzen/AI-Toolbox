#ifndef AI_TOOLBOX_IMPL_SEEDER_HEADER_FILE
#define AI_TOOLBOX_IMPL_SEEDER_HEADER_FILE

#include <random>

namespace AIToolbox::Impl {
    /**
     * @brief This class is an internal class used to seed all random engines in the library.
     *
     * To avoid seeding all generators with a single seed equal to the current time, only
     * this class is setup with the time seed, while all others are seeded with numbers
     * generated from this class to obtain maximum randomness.
     */
    class Seeder {
        public:
            static unsigned getSeed();
        private:
            Seeder();

            std::default_random_engine generator_;
    };
}

#endif
