"""
Testing saved models within within melee environment.
"""

import argparse
import torch
import numpy as np
from melee import enums
from parameters import MAX_STEP, STATE_DIM, ACTION_DIM

# from observation_normalizer import ObservationNormalizer
from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import PPOAgent, NOOP, CPU


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name", type=str, default="melee", help="name of environement"
)
parser.add_argument(
    "--model_path", type=str, default="./models/", help="where models are saved"
)
parser.add_argument(
    "--episode_num", type=int, default=10, help="How many times to test"
)
parser.add_argument(
    "--iso",
    type=str,
    default="../ssbm.iso",
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO",
)
args = parser.parse_args()


def run():
    """
    Start testing with given options.
    """

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(device)

    torch.manual_seed(500)
    np.random.seed(500)

    # players = [PPOAgent(enums.Character.FOX, device), NOOP(enums.Character.FOX)]
    players = [
        PPOAgent(enums.Character.FOX, 1, 2, device, STATE_DIM, ACTION_DIM),
        PPOAgent(enums.Character.FOX, 2, 1, device, STATE_DIM, ACTION_DIM),
    ]
    # players = [NOOP(enums.Character.FOX), NOOP(enums.Character.FOX)]

    # normalizer = ObservationNormalizer(s_dim)
    players[0].ppo.actor_net = torch.load(args.model_path + "actor_net.pt")
    players[0].ppo.critic_net = torch.load(args.model_path + "critic_net.pt")
    players[1].ppo.actor_net = torch.load(args.model_path + "actor_net.pt")
    players[1].ppo.critic_net = torch.load(args.model_path + "critic_net.pt")
    # normalizer.load(args.model_path)

    for episode_id in range(args.episode_num):
        score = 0

        env = MeleeEnv(args.iso, players, fast_forward=True)
        env.start()
        now_s, _ = env.reset(enums.Stage.FINAL_DESTINATION)

        action_pair = [0, 0]
        for step_cnt in range(MAX_STEP):
            if step_cnt > 100:

                action_pair[0], _ = players[0].act(now_s)
                action_pair[1], _ = players[1].act(now_s)

                next_s, r, done, _ = env.step(*action_pair)
                # next_state = normalizer(next_state)

                score += r[0]
                now_s = next_s

                if done:
                    break
            else:
                action_pair = [0, 0]
                now_s, _, _, _ = env.step(*action_pair)

        env.close()

        print("episode: ", episode_id, "\tscore: ", score)


if __name__ == "__main__":
    run()
