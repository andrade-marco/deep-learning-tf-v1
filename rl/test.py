import gym

env = gym.make("CartPole-v1", render_mode="human")

env_info = env.reset()
observation = env_info[0]
print(f"INITIAL OBSERVATION: {observation}")

for t in range(1000):
    env.render()

    cart_pos = observation[0]
    cart_vel = observation[1]
    pole_ang = observation[2]
    ang_vel = observation[3]

    if pole_ang > 0:
        action = 1
    else:
        action = 0

    observation, reward, terminated, truncated, info = env.step(action)

    print(f"************ STEP {t + 1} *****************")
    print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")
