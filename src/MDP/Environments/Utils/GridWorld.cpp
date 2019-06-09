#include <AIToolbox/MDP/Environments/Utils/GridWorld.hpp>

#include <algorithm>
#include <cassert>

namespace AIToolbox::MDP {
    GridWorld::State::State(int xx, int yy, size_t ss) :
        x(xx), y(yy), s(ss) {}

    GridWorld::State::operator size_t() { return s; }

    GridWorld::GridWorld(unsigned w, unsigned h) :
            width_(w), height_(h)
    {
        //assert(x > 0);
        //assert(y > 0);
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

    GridWorld::State GridWorld::operator()(int x, int y) const {
        x = boundX(x);
        y = boundY(y);
        return State(x, y, static_cast<size_t>(x + y*width_));
    }

    GridWorld::State GridWorld::operator()(const size_t s) const {
        return State(
            std::min((int)s%width_, width_-1),
            std::min((int)s/width_, height_-1),
            s
        );
    }

    int GridWorld::boundX(int x) const {
        if ( x < 0 ) return 0;
        if ( x >= (int)width_ ) return width_ - 1;
        return x;
    }

    int GridWorld::boundY(int y) const {
        if ( y < 0 ) return 0;
        if ( y >= (int)height_ ) return height_ - 1;
        return y;
    }

    unsigned GridWorld::getWidth() const { return width_; }
    unsigned GridWorld::getHeight() const { return height_; }
}
