#ifndef AI_TOOLBOX_FACTORED_MDP_TOROIDAL_WORLD_STATE
#define AI_TOOLBOX_FACTORED_MDP_TOROIDAL_WORLD_STATE

#include <algorithm>

enum Direction { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, STAND = 4 };

class ToroidalWorldState {
    public:
        ToroidalWorldState(unsigned sx, unsigned sy, int x, int y) : MAP_SIZE_X(sx), MAP_SIZE_Y(sy) { setX(x); setY(y); }

        operator size_t() { return x_ + y_*MAP_SIZE_X; }

        void setAdjacent(Direction d) {
            switch ( d ) {
                case UP:    setY(y_-1); return;
                case DOWN:  setY(y_+1); return;
                case LEFT:  setX(x_-1); return;
                case RIGHT: setX(x_+1); return;
                default: return;
            }
        }
        void setX(int newX) {
            x_ = (newX % MAP_SIZE_X + MAP_SIZE_X) % MAP_SIZE_X;
        }
        void setY(int newY) {
            y_ = (newY % MAP_SIZE_Y + MAP_SIZE_Y) % MAP_SIZE_Y;
        }
        unsigned getX() const { return x_; }
        unsigned getY() const { return y_; }

    private:
        unsigned MAP_SIZE_X, MAP_SIZE_Y;

        unsigned x_, y_;
};

#endif
