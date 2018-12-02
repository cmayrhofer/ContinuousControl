# Report
-------------

In the following we provide a detailed write up of the solution of the _single reacher_.
We have implemented a ddpg-agent, i.e. deep deep deterministic policy gradient agent, similar to the one outlined and implemented in [Lillicrap, Hunt, et. al.](http://arxiv.org/abs/1509.02971).
The idea behind DDPG is to leverage the  successful ideas behind deep q-learning networks DQN [Mnih, Kavukcuoglu, Silver, et. al.](http://www.nature.com/articles/nature14236) to problems with a continuous action space. The problem with the pure DQN approach and continuous action spaces is that one would need at (almost) all steps an optimization process which is computationally not feasible.
To circumvent this problem, an additional neural network is introduced, i.e. the actor, which learns to approximate optimal action directly. The Q-network is still of importance because it serves as a guidance, i.e. a critic, for the updates of the policy (actor) network.

As the authors of the DDPG paper mentioned the following two key features of the DQN approach are still crucial:
* memory replay: randomization over the data to remove correlations in the observation sequences and to be data efficient;
* target networks: the networks are trained with target networks to give a consistent target during temporal learning (to avoid dangerous correlations); The updates of the target networks are done softly, i.e. the weights of the actual actor and critic networks are slowly propagated to the targets, cf. the pseudo code below. This is different to the original approach for DQN where the update was a actual copying of weights but only every some thousand steps.

In addition to those the authors of DDPG add:
* batch normalization [Ioffe and Szegedy](http://arxiv.org/abs/1502.03167).

Next, we describe in detail how the learning is done and implemented
in our dqn agent.

## The Learning Algorithm

The most convenient way to give the learning algorithm is in terms of pseudocode:

**Algorithm for deep deterministic policy gradient approach with experience replay and soft update.**:

Initialize critic network (action-value function) $Q(s, a|\theta^Q)$ and actor $\mu(s|\theta^\mu)$ with random weights $\theta^Q$ and $\theta^\mu$.

Initialize target networks $\hat Q$ and $\hat \mu$ with weights $\theta^{\hat Q} \leftarrow \theta^Q$, $\theta^{\hat \mu} \leftarrow \theta^\mu$

Initialize replay memory $D$ to capacity ReplayBufferSize

**For** episode=1, MaxEpisodes **do**
>  Initialize/reset environment and get first state $s_1$
>
> Initialize a random process (noise) $\mathcal N$ for action exploration
>
> **For** t=1, T **do**
>> Select action $a_t = \mu(s_t|\theta^\mu) + \mathcal N_t$ according to the current policy and exploration noise
>>
>> Execute action $a_t$ and observe reward $r_t$ and observe new state $s_{t+1}$
>>
>> Store transition $(s_t,\, a_t,\, r_t,\, s_{t+1})$ in $D$
>>
>> Sample a random minibatch of $N$ transitions $(s_i,\,a_i,\,r_i,\,s_{i+1})$ from $D$
>>
>> Set $y_i = r_i + \gamma\, \hat Q(s_{i+1},\,\hat \mu(s_{i+1}|\theta^{\hat\mu})| \theta^{\hat Q})
>>
>> Update critic by minimizing the loss: $L = \tfrac1N\sum_i(y_i - Q(s_i,\, a_i|\theta^Q))^2
>>
>> Update the actor policy using the sampled policy gradient:
>>
>> $$ \nabla_{\theta^\mu} J \approx \frac1N \sum_i\nabla_a Q(s,\,a|\theta^Q)|_{s=s_i,\,a=\mu(s_i)}\nabla_{\theta^\mu}\mu(s|\theta^\mu)|_{s_i}$$
>> 
>> Update the target networks (soft update):
>>
>> $$ \theta^{\hat Q} \leftarrow \tau \theta^Q + (1 - \tau)\theta^{\hat Q} $$
>>
>> $$ \theta^{\hat \mu} \leftarrow \tau \theta^\mu + (1 - \tau)\theta^{\hat\mu}$$
>>
> **End For**

**End For**

The following table summarizes the values for all the parameters which are used during training:

|ReplayBufferSize| BatchSize |Gamma | Learning Rate Actor| Learning Rate Critic| $\tau$ | MaxEpisodes | Weight Decay | 
|------------------|---------|------|--------------------|--------------------|-----|------|---|
| 100000           | 128     | 0.99 |      0.0002        |      0.0002        |0.001|1000 | 0.0001 |

For the exploratory noise the parameters were:

| $\mu$ | $\theta$ | $\sigma$ |
|-------|----------|----------|
|0.     |   0.15   |    0.1   | 



The implementation of the above algorithm was done in a function called `ddpg`. This function makes us of five classes: `Agent`, `OUNoise`, `ReplayBuffer`, `Actor` and `Critic`.
* The `ReplayBuffer` class has the functions: `add` and `sample`;
* The `OUNoise` class has the functions: `reset` and `sample`;
* `Actor` and `Critic`: these classes define the DNN which approximate the optimal policy (function) and the action-value function, respectively, and are described in more detail below;
* The `Agent` class has the functions: `step`, `act`, `learn`, and `soft_update` where `act` gives $a_t$ and `step` calls, as in the order of the pseudocode,  `add`, `sample`, `learn`, and `soft_update`.


To see the performance increase of our agent we track the score for all episodes he is playing. This scores are the return values of the `ddpg` function.


### Learning Curve

![Learning Curve](learning_curve.png)
In this graphic we see the learning performance of our algorithm. We note that a mean score of 32 (over 100 consecutive episodes) is reached around 270 episodes.

## The Architecture of the Actor and Critic Network

The neural network which learns to approximate the optimal policy function consists of three linear fully connected layers. Between every two layers we use rectifier as non-linear activation functions. After the first fully connected layer we add a batch normalization. And in the very end a tanh is used to constrain the action to the action space which goes from $-1$ to $1$. In more detail, the network looks as follows:

* The first fully connected layer has 33 input-channels, for the 33 dimensional state vector, and 265 output channels.

* First ReLU layer.

* Batch normalization

* The hidden fully connected layer with 256 input and 128 output channels, respectively.

* Second ReLU layer.

* Output layer with 128 input and 4 output channels, where we act with a tanh on each output channel.


The neural network which learns to approximate the action-value function consists of four linear fully connected layers. Between every two layers we use rectifier as non-linear activation functions. After the first fully connected layer we add a batch normalization and concatenate the output of the actor DNN. In more detail, the network looks as follows:

* The first fully connected layer has 33 input-channels, for the 33 dimensional state vector, and 265 output channels.

* First ReLU layer.

* Batch normalization

* The hidden fully connected layer with 260 input and 256 output channels, respectively.

* Second ReLU layer.

* The hidden fully connected layer with 256 input and 128 output channels, respectively.

* Third ReLU layer.

* Output layer with 128 input and 1 output channel.




## Ideas for Future Work

* Improve the learning performance of the agent by implementing a [prioritized experience replay](https://arxiv.org/abs/1511.05952).

* Play around with the parameters and the network architectures to see what the 'minimal' DDPG configuration is which solves this environment. This is especially interesting because before adding batch normalization the learning would not take off and it would be good to know whether by tuning the parameters or changing the DNNs the agent could solve the environment also without batch normalization.

* Try to solve the version of the environment with 20 agents but by using either A3C, A2C, or [D4PG](http://arxiv.org/abs/1804.08617). This would be especially interesting because for the ddpg-agent we could reuse the code from the [pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) and [bipedalWalker](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) solutions. However, if we would use a different approach that wouldn't be the case anymore.

* Go beyond the reacher environment and try to solve the Crawler environment.

