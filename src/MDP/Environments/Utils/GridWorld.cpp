#include <AIToolbox/MDP/Environments/Utils/GridWorld.hpp>

#include <algorithm>
#include <cassert>
#include <cstdlib>

namespace AIToolbox::MDP {
    GridWorld::State::State(int xx, int yy, size_t ss) :
        x(xx), y(yy), s(ss) {}

    GridWorld::State::operator size_t() { return s; }

    bool GridWorld::State::operator==(const State & other) const {
        return s == other.s;
    }

    GridWorld::GridWorld(unsigned w, unsigned h, bool torus) :
            width_(w), height_(h), isTorus_(torus)
    {
        assert(width_ > 0);
        assert(height_ > 0);
    }

    GridWorld::State GridWorld::getAdjacent(Direction d, State s) const {
        using namespace GridWorldEnums;
        switch ( d ) {
            case UP:    return operator()(s.x, s.y-1);
            case DOWN:  return operator()(s.x, s.y+1);
            case LEFT:  return operator()(s.x-1, s.y);
            default:    return operator()(s.x+1, s.y);
        }
    }

    GridWorld::State GridWorld::getAdjacent(size_t d, State s) const {
        return getAdjacent((GridWorldEnums::Direction)d, s);
    }

    unsigned GridWorld::distance(const State & s1, const State & s2) const {
        if (!isTorus_)
            return std::abs(s1.x - s2.x) + std::abs(s1.y - s2.y);

        return
            std::min(std::abs(s1.x - s2.x), static_cast<int>(width_) - std::abs(s1.x - s2.x)) +
            std::min(std::abs(s1.y - s2.y), static_cast<int>(height_) - std::abs(s1.y - s2.y));
    }

    GridWorld::State GridWorld::operator()(int x, int y) const {
        x = boundX(x);
        y = boundY(y);
        return State(x, y, static_cast<size_t>(x + y*width_));
    }

    GridWorld::State GridWorld::operator()(const size_t s) const {
        assert(s < getS());
        return State(
            std::min((int)s%width_, width_-1),
            std::min((int)s/width_, height_-1),
            s
        );
    }

    int GridWorld::boundX(int x) const {
        if (isTorus_) {
            while (x < 0) x += width_;
            return x % width_;
        }
        if ( x < 0 ) return 0;
        if ( x >= (int)width_ ) return width_ - 1;
        return x;
    }

    int GridWorld::boundY(int y) const {
        if (isTorus_) {
            while (y < 0) y += height_;
            return y % width_;
        }
        if ( y < 0 ) return 0;
        if ( y >= (int)height_ ) return height_ - 1;
        return y;
    }

    unsigned GridWorld::getWidth() const { return width_; }
    unsigned GridWorld::getHeight() const { return height_; }
    size_t GridWorld::getS() const { return width_ * height_; }
}
