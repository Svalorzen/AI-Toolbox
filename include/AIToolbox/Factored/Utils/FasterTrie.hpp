#include <random>
#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored {
    class FasterTrie {
        public:
            FasterTrie(Factors f);
            size_t insert(PartialFactors pf);
            void erase(size_t id, const PartialFactors & pf);
            std::vector<size_t> filter(const Factors & f) const;
            std::pair<std::vector<size_t>, Factors> reconstruct(const PartialFactors & pf) const;
            size_t size() const;
            const Factors & getF() const;
        private:
            Factors F;
            size_t counter_;

            std::minstd_rand engine;
            std::vector<std::vector<std::vector<std::pair<size_t, PartialFactors>>>> keys_;
            mutable std::vector<unsigned char> taken_; // Booleans
    };
}
