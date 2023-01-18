from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class WallGapEnv(MiniGridEnv):
    """
    Environment with one wall with a small gap to cross through.
    """

    def __init__(self, size, obstacle_type=Wall, seed=None):
        self.obstacle_type = obstacle_type
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        # self.agent_pos = (1, 1)
        # self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        # Generate and store random gap position
        self.gap_pos = np.array((
            self._rand_int(2, width - 2),
            self._rand_int(1, height - 1),
        ))

        # Place the obstacle wall
        self.grid.vert_wall(self.gap_pos[0], 1, height - 2, self.obstacle_type)

        # Put a hole in the wall
        self.grid.set(*self.gap_pos, None)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(self.gap_pos[0], height))        

        self.mission = "find the opening and get to the green goal square"

class WallGapS5Env(WallGapEnv):
    def __init__(self):
        super().__init__(size=5)

class WallGapS6Env(WallGapEnv):
    def __init__(self):
        super().__init__(size=6)

class WallGapS7Env(WallGapEnv):
    def __init__(self):
        super().__init__(size=7)

register(
    id='MiniGrid-WallGapS5-v0',
    entry_point='gym_minigrid.envs:WallGapS5Env'
)

register(
    id='MiniGrid-WallGapS6-v0',
    entry_point='gym_minigrid.envs:WallGapS6Env'
)

register(
    id='MiniGrid-WallGapS7-v0',
    entry_point='gym_minigrid.envs:WallGapS7Env'
)
