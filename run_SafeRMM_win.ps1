# Start-Sleep -Seconds 15

# train Safe-RMM in Crossing scenario

python -u run_ppo_carla.py --env_name "CARLA" --scenario_name crossing  --crossing_n 3 --algorithm_name "wmappo" --experiment_name "CAV_train" --seed 0 -c 5 --num_agents 3 --discretize 5 --lr 2e-4 --critic_lr 2e-4 --episode_length 240 --episodes 5 --share_policy --use_recurrent_policy --use_centralized_V --spec_view birdeye --ucv_fs_step 66 --avg_coef 0.2 > logs/train_safe_rmm_crossing.out 2> logs/train_safe_rmm_crossing.err 

# train Safe-RMM in Highway scenario

# python -u run_ppo_carla.py --env_name "CARLA" --scenario_name highway --scene_n 0 --algorithm_name "wmappo" --experiment_name "CAV_train" --seed 0 -c 6 --num_agents 3 --discretize 5 --lr 2e-4 --critic_lr 2e-4 --episode_length 300 --avg_coef 0.2 --share_policy --episodes 4 > logs/train_safe_rmm_highway.out 2> logs/train_safe_rmm_highway.err 

# Start-Sleep -Seconds 15

# train Safe-MM (non-robust policy) in Crossing scenario

# python -u run_ppo_carla.py --env_name "CARLA" --scenario_name crossing --algorithm_name "rmappo" --experiment_name "CAV_train" --seed 0 -c 5 --num_agents 3 --discretize 5 --lr 2e-4 --critic_lr 2e-4 --episode_length 240 --episodes 5 --share_policy --use_recurrent_policy --use_centralized_V --spec_view birdeye --ucv_fs_step 66 --avg_coef 0.2 --crossing_n 3 > logs/train_safe_mm_crossing.out 2> logs/train_safe_mm_crossing.err 
