#include <AIToolbox/Factored/MDP/Environments/TigerAntelope.hpp>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox::Factored::MDP {
    TigerAntelope::TigerAntelope(unsigned width, unsigned height) :
            grid_(width, height, true), rand_(Impl::Seeder::getSeed())
    {
        antelopePosition_ = grid_(width / 2, height / 2);
    }

    std::tuple<State, Rewards> TigerAntelope::sampleSRs(const State & s, const Action & a) const {
        std::tuple<State, Rewards> retval;
        auto & [s1, rews] = retval;

        // Default values.
        s1 = s;
        rews.resize(2);
        rews << -0.5, -0.5;

        unsigned moved = 0;
        for (size_t i = 0; i < 2; ++i) {
            if (a[i] != 4) {
                ++moved;
                s1[i] = grid_.getAdjacent(a[i], grid_(s[i]));
            }
        }
        // Check if collided.
        if (s1[0] == s1[1]) {
            rews << -50.0, -50.0;
            std::uniform_int_distribution<size_t> d(0, grid_.getS() - 1);
            do {
                s1[0] = d(rand_);
            } while (s1[0] == antelopePosition_);
            do {
                s1[1] = d(rand_);
            } while (s1[1] == antelopePosition_ || s1[1] == s1[0]);
        } else if (moved > 0) {
            // Check if one moved to antelope
            if (s1[0] == antelopePosition_ || s1[1] == antelopePosition_) {
                auto thisId  = s1[0] == antelopePosition_ ? 0 : 1;
                auto otherId = s1[0] == antelopePosition_ ? 1 : 0;

                bool support = false;
                for (auto d : AIToolbox::MDP::GridWorldEnums::Directions)
                    if (s1[otherId] == grid_.getAdjacent(d, grid_(antelopePosition_)))
                        support = true;

                // If only one captured the antelope, we succeeded.
                if (support && moved == 1) {
                    // Captured!
                    rews << 37.5, 37.5;
                    return retval;
                } else {
                    rews[thisId] = -5.0;
                    std::uniform_int_distribution<size_t> d(0, grid_.getS() - 1);
                    do {
                        s1[thisId] = d(rand_);
                    } while (s1[thisId] == antelopePosition_ || s1[thisId] == s1[otherId]);
                }
            }
        }

        // "Move" the antelope now; we actually decide which direction it
        // moves, and if so we shift the whole world around.
        std::bernoulli_distribution d(0.2);
        if (!d(rand_)) {
            std::vector<size_t> goodDirections;
            for (auto d : AIToolbox::MDP::GridWorldEnums::Directions) {
                auto adjacent = grid_.getAdjacent(d, grid_(antelopePosition_));
                if (adjacent != s1[0] && adjacent != s1[1])
                    goodDirections.push_back(d);
            }
            std::uniform_int_distribution<size_t> d(0, goodDirections.size() - 1);
            // Shift both tigers in the opposite direction to mimic the
            // antelope moving.
            auto dir = (goodDirections[d(rand_)] + 2) % 4;
            s1[0] = grid_.getAdjacent(dir, grid_(s1[0]));
            s1[1] = grid_.getAdjacent(dir, grid_(s1[1]));
        }
        return retval;
    }

    bool TigerAntelope::isTerminalState(const State & s) const {
        if (!(s[0] == antelopePosition_ || s[1] == antelopePosition_))
            return false;

        // If the other tiger is next to the antelope then we have captured it.
        // Note that it is not possible to end up in this situation by having
        // both agents move at the same time, as the sampleSRs code will make
        // sure that the unsupported tiger is teleported somewhere else.
        auto otherId = s[0] == antelopePosition_ ? 1 : 0;
        for (auto d : AIToolbox::MDP::GridWorldEnums::Directions)
            if (s[otherId] == grid_.getAdjacent(d, grid_(antelopePosition_)))
                return true;

        return false;
    }

    State TigerAntelope::getS() const {
        State s(2, grid_.getS());
        return s;
    }

    Action TigerAntelope::getA() const {
        Action a(2, 5);
        return a;
    }

    size_t TigerAntelope::getAntelopeState() const { return antelopePosition_; }
    double TigerAntelope::getDiscount() const { return 0.9; }
    const AIToolbox::MDP::GridWorld & TigerAntelope::getGrid() const { return grid_; }

    std::string TigerAntelope::printState(const State & s) const {
        std::string retval;
        for (size_t y = 0; y < grid_.getHeight(); ++y) {
            for (size_t x = 0; x < grid_.getWidth(); ++x) {
                auto cell = grid_(x, y);
                if (cell == s[0]) {
                    retval += "t1";
                } else if (cell == s[1]) {
                    retval += "t2";
                } else if (cell == antelopePosition_) {
                    retval += "aa";
                } else {
                    retval += "..";
                }
                retval += "  ";
            }
            retval += '\n';
        }
        return retval;
    };
}
