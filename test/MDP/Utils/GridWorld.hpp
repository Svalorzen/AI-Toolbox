#ifndef AI_TOOLBOX_MDP_GRIDWORLD_STATE
#define AI_TOOLBOX_MDP_GRIDWORLD_STATE

#include <algorithm>

enum Direction { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };

class GridWorldState {
    public:

        GridWorldState(int sx, int sy, int x, int y) : MAP_SIZE_X(sx), MAP_SIZE_Y(sy) { setX(x); setY(y); }
        GridWorldState(int sx, int sy, size_t s) : MAP_SIZE_X(sx), MAP_SIZE_Y(sy),
                                                   x_(std::min((int)s%MAP_SIZE_X, MAP_SIZE_X-1)), 
                                                   y_(std::min((int)s/MAP_SIZE_X, MAP_SIZE_Y-1)) {}

        operator size_t() { return x_ + y_*MAP_SIZE_X; }

        void setAdjacent(Direction d) {
            switch ( d ) {
                case UP:    setY(y_-1); return;
                case DOWN:  setY(y_+1); return;
                case LEFT:  setX(x_-1); return;
                case RIGHT: setX(x_+1); return;
            }
        }
        void setX(int newX) {
            if ( newX < 0 ) x_ = 0;
            else if ( newX >= MAP_SIZE_X ) x_ = MAP_SIZE_X - 1;
            else x_ = newX;
        }
        void setY(int newY) {
            if ( newY < 0 ) y_ = 0;
            else if ( newY >= MAP_SIZE_Y ) y_ = MAP_SIZE_Y - 1;
            else y_ = newY;
        }
        int getX() const { return x_; }
        int getY() const { return y_; }

    private:
        int MAP_SIZE_X, MAP_SIZE_Y;

        int x_, y_;
};

class GridWorld {
    public:
        GridWorld(size_t x, size_t y) : MAP_SIZE_X(x), MAP_SIZE_Y(y) {}

        GridWorldState operator()(size_t x, size_t y) const {
            return GridWorldState(MAP_SIZE_X, MAP_SIZE_Y, x, y); 
        }

        GridWorldState operator()(size_t s) const {
            return GridWorldState(MAP_SIZE_X, MAP_SIZE_Y, s); 
        }

        size_t getSizeX() const { return MAP_SIZE_X; }
        size_t getSizeY() const { return MAP_SIZE_Y; }

    private:
        int MAP_SIZE_X, MAP_SIZE_Y;
};


#endif
