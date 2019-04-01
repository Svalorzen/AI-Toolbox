#include <AIToolbox/Factored/Utils/FasterTrie.hpp>

namespace AIToolbox::Factored {
    FasterTrie::FasterTrie(Factors f) : F(std::move(f)), taken_(F.size()) {}

    size_t FasterTrie::insert(PartialFactors pf) {
        keys_[pf.first[0]][pf.second[0]].emplace_back(counter_, std::move(pf));
        return counter_++;
    }

    void FasterTrie::erase(size_t id, const PartialFactors & pf) {
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
            for (size_t i = 1; i < (j + 1) && i < pf.first.size(); ++i) {
                if (pf.first[i] > f.size())
                    return true;
                if (f[pf.first[i]] != pf.second[i])
                    return false;
            }
            return true;
        };

        size_t i = 0;
        for (; i < f.size(); ++i) {
            for (const auto & [id, pf] : keys_[i][f[i]]) {
                if (matchPartial(f, pf, f.size() - i))
                    retval.push_back(id);
            }
        }
        // We also match to everybody after this point.
        for (; i < keys_.size(); ++i)
            for (const auto & keys : keys_[i])
                for (const auto & id_pf : keys)
                    retval.push_back(id_pf.first);

        return retval;
    }

    std::pair<std::vector<size_t>, Factors> FasterTrie::reconstruct(const PartialFactors & pf) const {
        std::pair<std::vector<size_t>, Factors> retval;
        auto & [ids, f] = retval;

        // Copy over set elements, and track them.
        for (size_t i = 0, j = 0; i < F.size(); ++i) {
            if (j == pf.first.size() || i < pf.first[j]) {
                taken_[i] = false;
            } else {
                taken_[i] = true;
                f[i] = pf.second[j];
                ++j;
            }
        }

        // Choice over factor id.
        for (size_t i = 0; i < keys_.size(); ++i) { // Randomize
            // Decide which factor values to iterate over. If the value is
            // known, we only look in the corresponding cell. Otherwise we look
            // at all of them.
            size_t j = 0;
            size_t jLimit = keys_[i].size();
            if (taken_[i]) {
                j = f[i];
                jLimit = j + 1;
            }
            // Choice over factor value.
            do { // Randomize
                // Choice over entry.
                for (size_t k = 0; k < keys_[i][j].size(); ++k) { // Randomize
                    auto & entry = keys_[i][j][k];
                    auto & entrypf = entry.second;
                    bool match = true;
                    for (size_t q = 0; q < entrypf.first.size(); ++q) {
                        if (taken_[entrypf.second[q]] && entrypf.second[q] != f[q]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        // We stop the outer loop here since we have now
                        // decided what value this factor has.
                        jLimit = j;
                        ids.push_back(entry.first);
                        for (size_t q = 0; q < entrypf.first.size(); ++q) {
                            taken_[q] = true;
                            f[q] = entrypf.second[q];
                        }
                    }
                }
            } while (++j < jLimit);
        }

        return retval;
    }
}
