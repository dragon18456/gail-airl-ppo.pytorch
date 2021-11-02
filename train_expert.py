import os
import argparse
from datetime import datetime
import torch
from torch import nn

from gail_airl_ppo.env import make_env, make_dmc_env
from gail_airl_ppo.algo import SAC
from gail_airl_ppo.trainer import Trainer
from gail_airl_ppo.network import GAILDiscrim, AIRLDiscrim


def run(args):
    if args.dmc:
        env = make_dmc_env(args.domain, args.task)
        env_test = make_dmc_env(args.domain, args.task)
    else:
        env = make_env(args.env_id)
        env_test = make_env(args.env_id)

    disc = None

    if args.disc_weights:
        if args.disc_type == "gail":
            disc = GAILDiscrim(
                state_shape=env.observation_space.shape,
                action_shape=env.action_space.shape,
                hidden_units=(100, 100),
                hidden_activation=nn.Tanh()
            ).to(torch.device("cuda" if args.cuda else "cpu"))
        else:
            disc = AIRLDiscrim(
                state_shape=env.observation_space.shape,
                gamma=.995,
                hidden_units_r=(100, 100),
                hidden_units_v=(100, 100),
                hidden_activation_r=nn.ReLU(inplace=True),
                hidden_activation_v=nn.ReLU(inplace=True)
            ).to(torch.device("cuda" if args.cuda else "cpu"))

        disc.load_state_dict(torch.load(args.disc_weights))
    
    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        disc=disc,
        disc_type=disc_type
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    if args.dmc:
        log_dir = os.path.join(
            'logs', f'{args.domain}-{args.task}', 'sac', f'seed{args.seed}-{time}')
    else:
        log_dir = os.path.join(
            'logs', args.env_id, 'sac', f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=10**6)
    p.add_argument('--eval_interval', type=int, default=10**4)
    p.add_argument('--dmc', action='store_true')
    p.add_argument('--domain', type=str, default='quadruped')
    p.add_argument('--task', type=str, default='walk')
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--disc_weights', type=str, default="")
    p.add_argument('--disc_type', type=str, default="gail")
    args = p.parse_args()
    run(args)
