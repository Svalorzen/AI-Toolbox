#ifndef AITOOLBOX_IMPL_SEEDER_HEADER_FILE
#define AITOOLBOX_IMPL_SEEDER_HEADER_FILE

#include <random>

namespace AIToolbox {
    namespace Impl {
        class Seeder {
            public:
                static unsigned getSeed();
            private:
                Seeder();

                std::default_random_engine generator_;
        };
    }
}

#endif
