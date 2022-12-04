import gym
import gym_minigrid
# env = gym.make('Pong-v0', render_mode='human')
# env.seed(42)
# obs = env.reset()
# done = False
# while not done:
#     _, _, done, _ = env.step(env.action_space.sample())
#     #env.render()
# env.close()

env = gym.make('SpaceInvaders-v0', render_mode='human')

env.seed(42)
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    _, _, done, _ = env.step(action)
    if done:
        env.reset()
env.close()

env = gym.make('MiniGrid-Empty-5x5-v0')

env.seed(42)
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    _, _, done, _ = env.step(action)
    if done:
        env.reset()
    env.render()
env.close()