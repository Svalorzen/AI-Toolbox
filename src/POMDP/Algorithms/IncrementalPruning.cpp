#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>

namespace AIToolbox {
    namespace POMDP {
        IncrementalPruning::IncrementalPruning(unsigned h) : horizon_(h) {}

        void IncrementalPruning::setHorizon(unsigned h) {
            horizon_ = h;
        }

        unsigned IncrementalPruning::getHorizon() const {
            return horizon_;
        }

        VList IncrementalPruning::crossSum(const VList & l1, const VList & l2, size_t a, size_t o) {
            VList c;

            if ( ! ( l1.size() && l2.size() ) ) return c;

            for ( auto & v1 : l1 ) {
                auto Obegin = std::begin(std::get<OBS>(v1));
                auto Oend   = std::end  (std::get<OBS>(v1));
                for ( auto & v2 : l2 ) {
                    // Cross sum
                    MDP::Values v(S, 0.0);
                    for ( size_t i = 0; i < S; ++i )
                        v[i] = std::get<VALUES>(v1)[i] + std::get<VALUES>(v2)[i];
                    // Here we can do a little trick. Since this function is only
                    // used in this class, we can safely assume that l1 is going to
                    // be the "bigger" list, while l2 is a list fresh out from the
                    // projection phase. Thus, at this point, l1 observations have
                    // o elements, and we are going to add the single one from
                    // l2, for a total of o+1.
                    VObs obs(o+1,0);
                    std::copy(Obegin, Oend, std::begin(obs));
                    obs[o] = std::get<OBS>(v2)[0];

                    c.emplace_back(std::move(v), a, std::move(obs));
                }
            }

            return c;
        }
    }
}
