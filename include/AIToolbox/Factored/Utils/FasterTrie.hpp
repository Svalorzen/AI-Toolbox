#include <random>
#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    class FasterTrie {
        public:
            using Entry = std::pair<size_t, PartialFactors>;
            using Entries = std::vector<Entry>;

            FasterTrie(Factors f);
            size_t insert(PartialFactors pf);
            void erase(size_t id, const PartialFactors & pf);
            // Not sorted.
            std::vector<size_t> filter(const Factors & f) const;

            // Not const because we shuffle, but nothign is actually changed.
            std::tuple<Entries, Factors, std::vector<unsigned char>> reconstruct(const PartialFactors & pf, bool remove = false);

            size_t size() const;
            const Factors & getF() const;
        private:
            Factors F;
            size_t counter_;

            std::vector<std::vector<Entries>> keys_;

            mutable std::ranlux24_base rand_; // Fastest engine possible, don't care about quality
            mutable std::vector<std::vector<size_t>> orders_;
    };
}
