import os
import argparse
import torch

from gail_airl_ppo.env import make_env, make_dmc_env
from gail_airl_ppo.algo import SACExpert
from gail_airl_ppo.utils import collect_demo


def run(args):
    if args.dmc:
        env = make_dmc_env(args.domain, args.task)
        env_test = make_dmc_env(args.domain, args.task)
    else:
        env = make_env(args.env_id)
        env_test = make_env(args.env_id)

    algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.weight
    )

    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )
    if args.dmc:
        buffer.save(os.path.join(
            'buffers',
            f'{args.domain}-{args.task}',
            f'size{args.buffer_size}_std{args.std}_prand{args.p_rand}.pth'
        ))
    else:
        buffer.save(os.path.join(
            'buffers',
            args.env_id,
            f'size{args.buffer_size}_std{args.std}_prand{args.p_rand}.pth'
        ))



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, required=True)
    p.add_argument('--dmc', action='store_true')
    p.add_argument('--domain', type=str, default='quadruped')
    p.add_argument('--task', type=str, default='walk')
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--buffer_size', type=int, default=10**6)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
