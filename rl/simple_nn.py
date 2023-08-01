import tensorflow as tf
import gym
import numpy as np


def main():
    num_inputs = 4
    num_hidden = 4
    num_outputs = 1  # prob of going left (1 - left = right)

    initializer = tf.contrib.layers.variance_scaling_initializer()

    X = tf.placeholder(tf.float32, shape=[None, num_inputs])

    hidden_layer_1 = tf.layers.dense(X, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
    hidden_layer_2 = tf.layers.dense(hidden_layer_1, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
    output_layer = tf.layers.dense(hidden_layer_2, num_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)

    probabilities = tf.concat(axis=1, values=[output_layer, 1-output_layer])
    action = tf.multinomial(probabilities, num_samples=1)

    init = tf.global_variables_initializer()

    episodes = 50
    step_limit = 500
    env = gym.make("CartPole-v1", render_mode="human")
    avg_steps = []

    with tf.Session() as session:
        init.run()

        for i_episode in range(episodes):
            obs = env.reset()[0]

            for step in range(step_limit):
                action_val = action.eval(feed_dict={X: obs.reshape(1, num_inputs)})
                obs, reward, terminated, truncated, info = env.step(action_val[0][0])  # 0 or 1

                if terminated or truncated:
                    avg_steps.append(step)
                    print(f"Done after {step} steps")
                    break

    print(f"After {episodes} episodes, average steps per game was {np.mean(avg_steps)}")
    env.close()


if __name__ == '__main__':
    main()


