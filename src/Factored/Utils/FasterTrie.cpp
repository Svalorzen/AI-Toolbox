#include <AIToolbox/Factored/Utils/FasterTrie.hpp>

#include <algorithm>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox::Factored {
    FasterTrie::FasterTrie(Factors f) :
            F(std::move(f)),
            counter_(0), keys_(F.size()),
            rand_(Impl::Seeder::getSeed()), orders_(F.size()+1)
    {
        for (size_t i = 0; i < F.size(); ++i)
            keys_[i].resize(F[i]);

        orders_[0].resize(F.size());
        std::iota(std::begin(orders_[0]), std::end(orders_[0]), 0);
        for (size_t i = 1; i < orders_.size(); ++i) {
            orders_[i].resize(F[i-1]);
            std::iota(std::begin(orders_[i]), std::end(orders_[i]), 0);
        }
    }

    size_t FasterTrie::insert(PartialFactors pf) {
        keys_[pf.first[0]][pf.second[0]].emplace_back(counter_, std::move(pf));
        return counter_++;
    }

    void FasterTrie::erase(const size_t id, const PartialFactors & pf) {
        // We don't care about ordering here.
        auto & keys = keys_[pf.first[0]][pf.second[0]];
        for (size_t i = 0; i < keys.size(); ++i) {
            if (id == keys[i].first) {
                std::swap(keys[i], keys.back());
                keys.pop_back();
                return;
            }
        }
    }

    std::vector<size_t> FasterTrie::filter(const Factors & f) const {
        std::vector<size_t> retval;

        auto matchPartial = [](const Factors & f, const PartialFactors & pf, const size_t j) {
            // We already know by definition that the first element will match.
            // We also know we can match at most j elements, so we bind that as well.
            for (size_t i = 1; i < j && i < pf.first.size(); ++i) {
                if (pf.first[i] >= f.size())
                    return true;
                if (f[pf.first[i]] != pf.second[i])
                    return false;
            }
            return true;
        };

        size_t i = 0;
        for (; i < f.size(); ++i)
            for (const auto & [id, pf] : keys_[i][f[i]])
                if (matchPartial(f, pf, f.size() - i))
                    retval.push_back(id);

        // We also match to everybody after this point.
        for (; i < keys_.size(); ++i)
            for (const auto & keys : keys_[i])
                for (const auto & id_pf : keys)
                    retval.push_back(id_pf.first);

        return retval;
    }

    std::tuple<FasterTrie::Entries, Factors> FasterTrie::reconstruct(const PartialFactors & pf, bool remove) {
        // Initialize retval
        std::tuple<Entries, Factors> retval;
        auto & [entries, f] = retval;
        f = F;

        // Copy over set elements, and track them.
        for (size_t i = 0; i < pf.first.size(); ++i)
            f[pf.first[i]] = pf.second[i];

        // We want to go over entries in the most randomized way possible
        // (although being fast is more important), as we want every possible
        // reconstruction to at least have a chance at being selected.

        // Random choice over factor ids.
        std::shuffle(std::begin(orders_[0]), std::end(orders_[0]), rand_);
        for (auto o : orders_[0]) {
            auto & keys = keys_[o];
            // Decide which factor values to iterate over. If the value is
            // known, we only look in the corresponding cell. Otherwise we look
            // at all of them (in a random order as well).
            size_t j = 0;
            bool done = false;
            decltype(&keys[0]) keysV;

            if (f[o] < F[o]) {
                done = true;
                keysV = &keys[f[o]];
            } else {
                std::shuffle(std::begin(orders_[o+1]), std::end(orders_[o+1]), rand_);
                keysV = &keys[orders_[o+1][0]];
            }

            do {
                // Finally, we go over all entries in this vector, and we
                // randomize them as well to be as fair as possible.
                std::shuffle(std::begin(*keysV), std::end(*keysV), rand_);
                for (size_t k = 0; k < keysV->size(); /* ++k later */) {
                    auto & entry = (*keysV)[k];
                    const auto & entrypf = entry.second;

                    // We increment k here to be able to go back if we remove the element.
                    ++k;

                    bool match = true;
                    for (size_t q = 0; q < entrypf.first.size(); ++q) {
                        const auto id = entrypf.first[q];
                        if (f[id] < F[id] && entrypf.second[q] != f[id]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        // We stop the outer loop here since we have now
                        // decided what value this factor has.
                        done = true;
                        for (size_t q = 0; q < entrypf.first.size(); ++q) {
                            const auto id = entrypf.first[q];
                            f[id] = entrypf.second[q];
                        }
                        if (remove) {
                            entries.emplace_back(std::move(entry));
                            entry = std::move(keysV->back());
                            keysV->pop_back();
                            --k;
                        } else {
                            entries.push_back(entry);
                        }
                    }
                }
                if (done || ++j >= orders_[o+1].size())
                    break;
                keysV = &keys[orders_[o+1][j]];
            } while (true);
        }
        return retval;
    }

    size_t FasterTrie::size() const {
        size_t retval = 0;
        for (const auto & keysF : keys_)
            for (const auto & keysV : keysF)
                retval += keysV.size();

        return retval;
    }

    const Factors & FasterTrie::getF() const { return F; }
}
