[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Continuous Control

In this repository, we provide a code implementation of a [D(eep) D(eterministic) P(olicy) G(radient) agent](http://arxiv.org/abs/1509.02971) which solves a modified version of the [Reacher](https://youtu.be/2N9EoF6pQyE) [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) environment of [unity](unity3d.com). The [modifications](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control) were provided by the [Udacity](https://eu.udacity.com/) team responsible for the [Deep Reinforcment Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). We refer to the modified Reacher environment with just one double-jointed arm as _single reacher_. 

![Trained Agent][image1]

The code solving the single reacher environment is split up into three parts:
* The `ddpg_agent.py`file provides the implementation of the (ddpg) `Agent` class, an `OUNoise` class to realise the Ornstein-Uhlenbeck process, and  a `ReplayBuffer` class.
* In `model.py` the deep neural networks which learn the policy (actor network) and the [action-value function](https://en.wikipedia.org/wiki/Reinforcement_learning) (critic network) are defined.
* The training of the agent, visuallisation of the agents performance during learning (i.e. learning curve), and the agent playing a test episode, after a successful training, is all done in the jupyter notebook `Continuous_Control.ipynb`. _All the necessary information on how to run the code are provided within this notebook._

Furthermore, the repository contains also the saved weights of a trained actor and critic. They can be found in the files `checkpoint_actor.pth` and `checkpoint_critic.pth`. Both pretrained weights files can be loaded via [pytorch](pytorch.org) to neural networks with the same architectures as the ones in `model.py`. In the `Report.md` file you can find further informations on the learning algorithm used to solve this environment and how it is implemented in the above listed files.


## Details of the RL Environment

As the above GIF-animation adumbrates, the agent has to move the end of his double-joined arm into the goal location, and keep it there. The task is episodic and we set the maximal amount of steps 10000 per episode.

* _`States`_: the state space is 26 dimensional and consists out of position, rotation, velocity, and angular velocities of the two arm Rigidbodies;
* _`Actions`_: the agent's action space is (continuous) 4 dimensional and corresponds to torque applicable to the two joints;
* _`Reward`_: the agent obtains a reward of $+0.1$ if the agent's hand is in the goal location, and $0$ else.

The environment is to be considered solved if the average score of the agent over 100 consecutive episodes is greater equal 30.

## Getting Started

To run the code provided in this repository, you must have python 3.6 or higher installed. In addition, you will need to have the packages: [numpy](http://www.numpy.org/), [matplotlib](https://matplotlib.org/), [torch](https://pytorch.org/) (including its dependencies) installed. Then follow the next three steps. Afterwards you should be able to run the `Navigation.ipynb` without any dependencies errrors.

1. Follow the steps in [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) to install the Unity ML-Agents Toolkit.
2. Download the single reacher environment from one of the links below. Select the environment which matches your operating system:
     - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
      - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
3. Place the file in the folder where you also placed the `ddpg_agent.py` file, the `model.py` file, and the `Continuous_Control.ipynb` notebook and unzip (or decompress) the file.

If you are still having issues after following this steps regarding the dependencies then please check out the more throughly configuration [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).


### Marginal Comment

As mentioned already above the single reacher environment is a modefication of the original reacher of Unity which was designed as a 'multiagent' test ground. The modifications where done by the [Udacity](https://eu.udacity.com/) team responsible for the [Deep Reinforcment Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program (to provide one option for the second of the three projects in the DRL Nanodegree program which have to be successfully complited in order to pass the course). 
Compared to standard single agent environments, as you can find them on [OpenAI-Gym](https://gym.openai.com/) for instance, we have to add the following two lines of code:
```
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```
This is due to fact that the single agent environment is derived from a 'multiagent' environment.

