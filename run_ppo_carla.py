import numpy as np
from pathlib import Path
import os
import time
import random
import torch
import sys
import setproctitle

from main.envs.carla.carla_environment import CarEnv
# from main.envs.carla.carla_environment_robust_CBF import CarEnv
# from main.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from configs.config_carla import get_config

def make_train_carla_env(args):
    if args.map=='Town05': # town05, 4 cars, 3 agents
        n_agents = args.cars-1
    elif args.map=='Town06':
        if args.cars > 9:
            raise RuntimeError("Wrong input for n_cars: {}".format(args.cars))
    
    assert args.num_agents > 0

    env = CarEnv(args=args, num_CAVs=args.num_agents)
    
    return env, args.num_agents

def parse_args(args, parser):
    #parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--exp_name", type=str, default="exp", help="name of experiment run")
    parser.add_argument("--benchmark", type=bool, default=False, help="benchmark")
    parser.add_argument("--render", default=False, action='store_true', help="render in window")
    parser.add_argument("--old", default=False, action='store_true', help="use old mpe")
    parser.add_argument("--num-episodes", type=int, default=100, help="number of episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="maximum steps in an episode")
    parser.add_argument("--entropy_reg", type=float, default=0.1, help="entropy coefficient in training actor")
    parser.add_argument("--left_lc", type=float, default=60, help="entropy coefficient in training actor")
    parser.add_argument("--right_lc", type=float, default=155, help="entropy coefficient in training actor")
    parser.add_argument("--timeout_lc", type=float, default=50, help="entropy coefficient in training actor")
    
    all_args = parser.parse_known_args(args)[0]
    return all_args

def run(args):
    #start = time()
    parser = get_config()
    all_args = parse_args(args, parser)
    print(all_args)
    

    if all_args.algorithm_name in ["wmappo", "maa2c"]:
        print("u are choosing to use wmappo, worst-case Q enabled")
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy
        print("u are choosing to use rmappo, RNN in policy")
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        # all_args.use_recurrent_policy = False 
    else:
        raise NotImplementedError

    #assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
    #    "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results") / all_args.env_name / \
            all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        raise NotImplementedError
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))
    
    # seed_time = int(time.time())
    seed_time = 0
    torch.manual_seed(all_args.seed + seed_time)
    torch.cuda.manual_seed_all(all_args.seed + seed_time)
    random.seed(all_args.seed + seed_time)
    np.random.seed(all_args.seed + seed_time)
    
    if all_args.run_rule_based_benchmark:
        from main.runner.shared.carla_runner_rule_based_share import CARLARunner as Runner
    elif all_args.share_policy:
        from main.runner.shared.carla_runner_shared import CARLARunner as Runner
    else:
        from main.runner.separated.carla_runner_general import CARLARunner as Runner

    # env init
    envs, num_agents = make_train_carla_env(all_args)
    all_args.base_state_length = envs.base_state_length
    all_args.max_num_car = envs.max_num_car

    config = {
        "all_args": all_args,
        "envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)

    if not all_args.test_only:
        runner.run()
    else:
        
        runner.eval()
    
    # post process
    envs.close()
    runner.save_as_pickle(os.path.join(run_dir, 'logging_dict.pkl'))

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()



if __name__ == '__main__':
    #arglist = parse_args()
    #print(arglist)
    run(sys.argv[1:])