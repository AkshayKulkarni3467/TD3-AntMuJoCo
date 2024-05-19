# TD3 Ant Walking Agent

## Overview:
This project trains the ant walker agent using TD3 algorithm.

## Overview of algorithm:
TD3 solves the overestimation bias of DDPG. TD3 is based on DDPG with three smart improvements:
- Additive clipped noise on actions
- Double critics and actors
- Delayed actors update
This addresses variance and the quality of the value function estimation.

## After Training GIF:
![after training gif 1](https://github.com/AkshayKulkarni3467/TD3-AntMuJoCo/assets/129979542/ad81fe81-2930-4e2e-9c5c-cb40a9b3dbdd)

![after training gif 2](https://github.com/AkshayKulkarni3467/TD3-AntMuJoCo/assets/129979542/80916b4f-a164-4a23-b665-d330eef6253e)

## Loss and Reward curves:
- Actor Loss
  
![td3_actor_loss](https://github.com/AkshayKulkarni3467/TD3-AntMuJoCo/assets/129979542/a055314d-8298-46e1-9de9-f99aa1734b47)

- Critic Loss

![td3_critic_loss](https://github.com/AkshayKulkarni3467/TD3-AntMuJoCo/assets/129979542/0ae62230-b57d-49fa-af3b-424a16c4e01b)

- Reward Curve

![td3_reward](https://github.com/AkshayKulkarni3467/TD3-AntMuJoCo/assets/129979542/67e33813-11c7-47f4-b308-fdc329d8f258)
