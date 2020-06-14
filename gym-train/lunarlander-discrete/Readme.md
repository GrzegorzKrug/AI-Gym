#
## Actor Critic Algorithm with connected nodes

There is no advantage function

This project proves, that we can solve proble with using 2 different models, that models `critic` and `actor` do not share any nodes except `input`.
Disadvantage to this is that both models detects different features from input, but agent model can be smaller.

![Nodes](./Model-39/model/actor.png) ![Nodes](./Model-39/model/critic.png)

### Log probabilty
![image](./action_probabilty.gif)


### First train

![image](./Model-39/scores-06-14--15-23-37.png)

#### A little smoothing
![image](./Model-39/scores-06-14--16-28-55.png)

### Landing gifs

##### Game 90
Agent is learning to fly

![Landing](./Model-39/replay-Model-39-90.gif)
##### Game 100
Agents is learning to land

![Landing](./Model-39/replay-Model-39-100.gif)
##### Game 120
Agent is optimizing trajectory

![Landing](./Model-39/replay-Model-39-120.gif)

##### Game 184
Agents is landing smoothly

![Landing](./Model-39/replay-Model-39-184.gif)

