import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.env import make_env, make_dmc_env
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer


def run(args):
    if args.dmc:
        env = make_dmc_env(args.domain, args.task)
        env_test = make_dmc_env(args.domain, args.task)
    else:
        env = make_env(args.env_id)
        env_test = make_env(args.env_id)
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=torch.device("cuda" if args.cuda else "cpu")
    )

    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length
    )

    if args.actor_weights:
        algo.actor.load_state_dict(torch.load(args.actor_weights))
    if args.critic_weights:
        algo.critic.load_state_dict(torch.load(args.critic_weights))

    time = datetime.now().strftime("%Y%m%d-%H%M")
    if args.dmc:
        log_dir = os.path.join(
            'logs', f"{args.domain}-{args.task}", args.algo, f'seed{args.seed}-{time}')
    else:
        log_dir = os.path.join(
            'logs', args.env_id, args.algo, f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10**7)
    p.add_argument('--eval_interval', type=int, default=10**5)
    p.add_argument('--dmc', action='store_true')
    p.add_argument('--domain', type=str, default='quadruped')
    p.add_argument('--task', type=str, default='walk')
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--actor_weights', type=str, default="")
    p.add_argument('--critic_weights', type=str, default="")
    args = p.parse_args()
    run(args)
