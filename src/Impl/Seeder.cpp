#include <AIToolbox/Impl/Seeder.hpp>

#include <chrono>
#include <limits>

namespace AIToolbox::Impl {
    Seeder Seeder::instance_;

    Seeder::Seeder() : generator_(std::chrono::system_clock::now().time_since_epoch().count()) {}

    unsigned Seeder::getSeed() {
        static std::uniform_int_distribution<unsigned> dist(0, std::numeric_limits<unsigned>::max());

        return dist(instance_.generator_);
    }

    void Seeder::setRootSeed(const unsigned seed) {
        instance_.generator_.seed(seed);
    }
}
