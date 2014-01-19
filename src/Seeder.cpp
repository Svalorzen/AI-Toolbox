#include "Seeder.hpp"

#include <chrono>
#include <limits>

namespace AIToolbox {
    namespace Impl {
        Seeder::Seeder() : generator_(std::chrono::system_clock::now().time_since_epoch().count()) {}

        unsigned Seeder::getSeed() {
            static Seeder instance;
            static std::uniform_int_distribution<unsigned> dist(0, std::numeric_limits<unsigned>::max());

            return dist(instance.generator_);
        }
    }
}
