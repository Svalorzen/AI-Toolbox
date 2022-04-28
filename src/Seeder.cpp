#include <AIToolbox/Seeder.hpp>

#include <chrono>
#include <limits>

namespace AIToolbox {
    Seeder Seeder::instance_;

    Seeder::Seeder() {
        rootSeed_ = std::chrono::system_clock::now().time_since_epoch().count();
        generator_.seed(rootSeed_);
    }

    unsigned Seeder::getSeed() {
        static std::uniform_int_distribution<unsigned> dist(0, std::numeric_limits<unsigned>::max());

        return dist(instance_.generator_);
    }

    void Seeder::setRootSeed(const unsigned seed) {
        instance_.rootSeed_ = seed;
        instance_.generator_.seed(instance_.rootSeed_);
    }

    unsigned Seeder::getRootSeed() {
        return instance_.rootSeed_;
    }
}
