#!/bin/bash

# nohup ~/Carla_915/CarlaUE4.sh -RenderOffScreen -quality-level=Low  -carla-rpc-port=2000 \
#     >> ~/nohup_carla.out 2>> ~/nohup_carla.err &
# sleep 10
# carla_pid=$(pidof CarlaUE4-Linux-Shipping)  

CUDA_VISIBLE_DEVICES=0 python -u run_ppo_carla.py --env_name "CARLA" --scenario_name highway --scene_n 0 --algorithm_name "wmappo" --experiment_name "CAV_train" --seed 0 -c 6 --num_agents 3 --discretize 5 --lr 2e-4 --critic_lr 2e-4 --episode_length 300 --avg_coef 0.2 --share_policy --episodes 4 > logs/train_safe_rmm_highway.out 2> logs/train_safe_rmm_highway.err

# exp_pid=$!
# sleep 10
# echo "Finished Carla PID: $carla_pid; Python PID: $exp_pid"
# kill -9 $carla_pid
# echo "Carla $carla_pid killed."
# sleep 10

