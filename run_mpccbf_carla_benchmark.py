import numpy as np
from pathlib import Path
import os
import time
import random
import torch
import sys
import setproctitle
from main.envs.carla.carla_environment_mpccbf import CarEnv
# from main.envs.carla.carla_environment_robust_CBF import CarEnv
# from main.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from configs.config_carla import get_config

def make_train_carla_env(args):
    if args.map=='Town05': # town05, 4 cars, 3 agents
        n_agents = args.cars-1
    elif args.map=='Town06':
        if args.cars in [5,6]: # town06, 5 cars, 3 agents
            n_agents = 3
        elif args.cars == 10: # town06, 10 cars, 5 agents
            n_agents = 5
        elif args.cars == 3: # special test scenario
            n_agents = 2
        elif args.cars == 4:
            n_agents = 2
        elif args.cars == 2:
            n_agents = 1
        else:
            raise RuntimeError("Wrong input for n_cars")
 

    if args.num_agents != 0:
        n_agents = args.num_agents

    # env = CarEnv(episode_length=args.episode_length, num_Vs=args.cars, num_CAVs=n_agents, \
    #              discretize=args.discretize, name_town=args.map, timestep=args.timestep, \
    #              safety_step=args.safety_step, no_render=args.no_render, host=args.host, \
    #              port=args.port, spec_view=args.spec_view, mute_lane_change=args.disable_lane_chg, \
    #              disable_CBF=args.disable_CBF, brief_state=args.brief_state, \
    #              normal_behave=args.normal_behave, remove_CBFA=args.remove_CBFA, \
    #              e_type=args.e_type, test_only=args.test_only, force_straight_step=args.fs_step, \
    #              cbf_robustness=args.cbf_robustness, debug=args.debug)

    env = CarEnv(args=args, num_CAVs=n_agents)
    
    return env, n_agents

def parse_args(args, parser):
    #parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    # parser.add_argument("--num_landmarks", type=int, default=3)
    # parser.add_argument('--num_agents', type=int, default=3, help="number of players")
    # parser.add_argument("--scenario", type=str, default="highway", help="name of the scenario script")
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
    all_args.scenario_name = 'highway'
    all_args.cars = 6
    all_args.discretize = 0
    all_args.episode_length = 500
    # all_args.spec_view = 'birdeye'

    if all_args.algorithm_name in ["rppo", "rmappo", "wmappo", "maa2c"]:
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name in ["mappo"]:
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
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
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
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

    # env init
    envs, num_agents = make_train_carla_env(all_args)
    #eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    #num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    
    from main.runner.separated.carla_runner_mpccbf import CARLARunner as Runner

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
