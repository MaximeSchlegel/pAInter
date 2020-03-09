import numpy as np


from painter.agents.Agent import Agent

from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy, SparseCategoricalCrossentropy, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


########################################################################################################################
class ActionValueModel(Model):

    def __init__(self,
                 action_size):
        super().__init__('mlp_policy')

        self.model_name = "FeatExtr:4_32f_Conv2D-" + \
                          "ActPick:3_128_64_128_Dense-" + \
                          "Val:3_128_128_128_Dense"

        self.action_size = action_size

        # TODO Defining learnable operators

        # Actor - Defining operators for policy network
        self.feat_extr_0 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2))
        self.feat_extr_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2))
        self.feat_extr_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2))
        self.feat_extr_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2))
        self.feat_extr_4 = GlobalAveragePooling2D()

        self.action_picker_1 = Dense(128, activation='relu')
        self.action_picker_2 = Dense(64, activation='relu')
        self.action_picker_3 = Dense(128, activation='relu')
        self.action_picker_4 = Dense(self.action_size, activation='relu')

        # Critic - Defining operators for value network
        self.value_1 = Dense(128, activation='relu')
        self.value_2 = Dense(128, activation='relu')
        self.value_3 = Dense(128, activation='relu')
        self.value_4 = Dense(1, name='value')

    def call(self,
             inputs):
        # Converting inputs array into a tensor
        # x = tf.convert_to_tensor(inputs)

        # inputs = [target_image, current_image]
        input_target, input_current = inputs

        # Extract the features of the target image
        latent_target = self.feat_extr_0(input_target)
        latent_target = self.feat_extr_1(latent_target)
        latent_target = self.feat_extr_2(latent_target)
        latent_target = self.feat_extr_3(latent_target)
        latent_target = self.feat_extr_4(latent_target)

        # Extract the feature of the current image
        latent_current = self.feat_extr_0(input_current)
        latent_current = self.feat_extr_1(latent_current)
        latent_current = self.feat_extr_2(latent_current)
        latent_current = self.feat_extr_3(latent_current)
        latent_current = self.feat_extr_4(latent_current)

        # Aggregate the fetaures of the two images
        latent_repr = Concatenate()([latent_target, latent_current])

        # Compare the latent representation and set the parameters of the action
        action = self.action_picker_1(latent_repr)
        action = self.action_picker_1(action)
        action = self.action_picker_1(action)
        action = self.action_picker_1(action)

        # Compute the evaluation of the current state
        value = self.value_1(latent_repr)
        value = self.value_2(value)
        value = self.value_3(value)
        value = self.value_4(value)

        # returning action actions and value
        return action, value

    def action_value(self,
                     obs):
        # Predicting action and value from observed state
        action, value = self.predict(obs)

        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)  # ?


########################################################################################################################
class A2CAgent(Agent):

    def __init__(self,
                 model,
                 gamma=0.99,
                 value_loss_factor=0.5,
                 entropy_loss_factor=0.0001,
                 learning_rate=0.0007):
        super().__init__()

        # Hyperparameters settings
        self.params = {"gamma": gamma,  # discount factor
                       "value_loss_factor": value_loss_factor,
                       "entropy_loss_factor": entropy_loss_factor,  # exploration factor
                       }

        # Defining and compiling model
        self.model = model
        self.model.compile(optimizer=RMSprop(lr=learning_rate),
                           loss=[self._logits_loss,
                                 self._value_loss,
                                 ]
                           )

    def train(self,
              environment,
              batch_size=32,
              updates=1000):

        # storage helpers for a single batch of data
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + environment.observation_space.shape)

        # memory leak tracker
        # tr = tracker.SummaryTracker()

        # Training loop
        episode_rewards = [
            0.0]  # cumulative episode reward (episode_rewards[i] contains the sum of rewards of episode i)
        next_obs = environment.reset()
        for update in range(updates):
            logging.info("\n\n    UPDATE: %03d \n\n" % (update))
            # Collecting batch of training data
            for step in range(batch_size):

                # Percepting the environment
                observations[step] = next_obs.copy()

                # Getting action and value predicted for this perception
                actions[step], values[step] = self.model.action_value(next_obs[None, :])

                # Updating the environment with action
                next_obs, rewards[step], dones[step], _ = environment.step(actions[step])

                # Updating current episode total reward
                episode_rewards[-1] += rewards[step]

                # If the episode has come to an end, restarting the environment and initializing cumulative reward for new episode
                if dones[step]:
                    episode_rewards.append(0.0)
                    next_obs = environment.reset()

                    # Monitoring
                    logging.info("    Episode: %03d, Reward: %03d" % (len(episode_rewards) - 1, episode_rewards[-2]))
                    logging.info("    actions: %s; rewards: %s; dones: %s; values: %s; observations: %s" % (
                    actions.shape, rewards.shape, dones.shape, values.shape, observations.shape))
                    logging.info("    episode_rewards: %s" % (len(episode_rewards)))
                    # tr.print_diff()
            # At that point, we have enough information to make a batch of data

            # Making batch
            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            # Training on collected batch
            acts_and_advs = np.concatenate([actions[:, None], advantages[:, None]],
                                           axis=-1)  # trick to include multiple arguments in a loss function
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])

            # Monitoring
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

        # tr.print_diff()
        return episode_rewards

    def decision(self,
                 perception):
        action, value = self.model.action_value(perception[None, :])
        return action

    def test(self,
             environment,
             render=False):
        obs, done, episode_reward = environment.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = environment.step(action)
            episode_reward += reward
            if render:
                environment.render()
        return episode_reward

    def _returns_advantages(self,
                            rewards,
                            dones,
                            values,
                            next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        # Calculating returns as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]

        # Calculating advantages
        advantages = returns - values

        return returns, advantages

    def _value_loss(self,
                    returns,
                    value):
        return self.params['value_loss_factor'] * mean_squared_error(returns,
                                                                     value)  # value loss is typically MSE between value estimates and returns

    def _logits_loss(self,
                     acts_and_advs,
                     logits):

        # Separating actions and advantages
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)

        # Calculating policy loss
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        weighted_sparse_ce = SparseCategoricalCrossentropy(
            from_logits=True)  # from_logits argument ensures transformation into normalized probabilities
        policy_loss = weighted_sparse_ce(actions, logits,
                                         sample_weight=advantages)  # policy loss is defined by policy gradients, weighted by advantages

        # Calculating entropy loss
        entropy_loss = categorical_crossentropy(logits, logits,
                                                from_logits=True)  # entropy loss can be calculated via CE over itself

        return policy_loss - self.params[
            'entropy_loss_factor'] * entropy_loss  # here signs are flipped because optimizer minimizes