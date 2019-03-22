#ifndef AI_TOOLBOX_MDP_GRIDWORLD
#define AI_TOOLBOX_MDP_GRIDWORLD

#include <cstddef>

namespace AIToolbox::MDP {
    /**
     * @brief This namespace exists in order to allow referencing the Direction values directly.
     */
    namespace GridWorldActions {
        /**
         * @brief The possible actions in a GridWorld-like environment.
         */
        enum Direction : size_t { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };
    }

    /**
     * @brief This class represents a simple rectangular gridworld.
     */
    class GridWorld {
        public:
            using Direction = GridWorldActions::Direction;
            struct State {
                operator size_t();
                private:
                    State(int xx, int yy, size_t ss);
                    int x, y;
                    size_t s;
                    friend GridWorld;
            };

            /**
             * @brief Basic constructor.
             *
             * @param width The number of columns in the world.
             * @param height The number of rows in the world.
             */
            GridWorld(unsigned width, unsigned height);

            /**
             * @brief This function returns the state next to the input in the chosen Direction.
             *
             * This function returns the same state if the Direction goes
             * outside the Grid boundaries.
             *
             * @param d The Direction to look for.
             * @param s The initial State.
             *
             * @return The State next to the input.
             */
            State getAdjacent(Direction d, State s) const;

            /**
             * @brief This function returns the State at the selected position.
             *
             * Coordinates are bound to the size of the GridWorld.
             *
             * @param x The x of the output State.
             * @param y The y of the output State.
             */
            State operator()(int x, int y) const;

            /**
             * @brief This function returns the State with the input numerical representation.
             *
             * @param s The State numerical representation.
             */
            State operator()(const size_t s) const;

            /**
             * @brief This function returns the width of the GridWorld.
             */
            unsigned getWidth() const;

            /**
             * @brief This function returns the height of the GridWorld.
             */
            unsigned getHeight() const;

        private:
            int boundX(int x) const;
            int boundY(int y) const;

            unsigned width_, height_;
    };
}

#endif
