from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class DoorKeyLavaEnv(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    def __init__(self, size, obstacle_type=Lava, seed=None):
        self.obstacle_type = obstacle_type
        super().__init__(
            grid_size=size,
            max_steps=10*size*size,
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
        self.agent_pos = (1, 1)
        self.agent_dir = 0

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
        #self.grid.set(*self.gap_pos, None)

        # Place a door in the wall
        #doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), *self.gap_pos)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(self.gap_pos[0], height)
        )

        self.mission = (
            "use the key to open the door and then avoid the lava and get to the goal"
            if self.obstacle_type == Lava
            else "find the door and get to the green goal square"
        )


class DoorKeyLavaEnv5x5(DoorKeyLavaEnv):
    def __init__(self):
        super().__init__(size=5)

class DoorKeyLavaEnv6x6(DoorKeyLavaEnv):
    def __init__(self):
        super().__init__(size=6)

class DoorKeyLavaEnv8x8(DoorKeyLavaEnv):
    def __init__(self):
        super().__init__(size=8)

class DoorKeyLavaEnv16x16(DoorKeyLavaEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-DoorKeyLava-5x5-v0',
    entry_point='gym_minigrid.envs:DoorKeyLavaEnv5x5'
)

register(
    id='MiniGrid-DoorKeyLava-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyLavaEnv6x6'
)

register(
    id='MiniGrid-DoorKeyLava-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyLavaEnv8x8'
)

register(
    id='MiniGrid-DoorKeyLava-16x16-v0',
    entry_point='gym_minigrid.envs:DoorKeyLavaEnv16x16'
)
