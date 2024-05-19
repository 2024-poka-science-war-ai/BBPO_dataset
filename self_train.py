"""
Start training models by PPO method within melee environment.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from melee import enums
from parameters import (
    MAX_STEP,
    CYCLE_NUM,
    MIN_TUPLES_IN_CYCLE,
    STATE_DIM,
    ACTION_DIM,
    DELAY,
)

# from observation_normalizer import ObservationNormalizer
from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import PPOAgent, NOOP, CPU
import random


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name", type=str, default="melee", help="name of environement"
)
parser.add_argument(
    "--continue_training",
    type=bool,
    default=False,
    help="whether to continue training with existing models",
)
parser.add_argument(
    "--model_path", type=str, default="./models/", help="where models are saved"
)
parser.add_argument(
    "--iso",
    type=str,
    default="../ssbm.iso",
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO",
)
parser.add_argument(
    "--print_fuck",
    type=bool,
    default=False,
    help="Whether to print logs to catch fucked up animations",
)

args = parser.parse_args()


def run():
    """
    Start training with given options.
    """

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(device)

    torch.manual_seed(500)
    np.random.seed(500)

    pl_lst = [
        PPOAgent(enums.Character.FOX, 1, 2, device, STATE_DIM, ACTION_DIM),
        PPOAgent(enums.Character.FOX, 2, 1, device, STATE_DIM, ACTION_DIM),
    ]

    episode_id = 0
    if args.continue_training:
        # load pre-trained models
        pl_lst[0].ppo.actor_net = torch.load(args.model_path + "actor_net.pt").to(
            device
        )
        pl_lst[0].ppo.critic_net = torch.load(args.model_path + "critic_net.pt").to(
            device
        )
        pl_lst[1].ppo.actor_net = torch.load(args.model_path + "actor_net.pt").to(
            device
        )
        pl_lst[1].ppo.critic_net = torch.load(args.model_path + "critic_net.pt").to(
            device
        )
        df = pd.read_csv("log_melee.csv")
        episode_id = len(df) - 1
    else:
        # clear log file
        with open("log_" + args.env_name + ".csv", "w", encoding="utf-8") as outfile:
            outfile.write("episode_id,score\n")

    for cycle_id in range(CYCLE_NUM):
        scores = []  # for log
        pl_lst[0].ppo.buffer.buffer.clear()  # PPO is an on-policy algorithm
        while pl_lst[0].ppo.buffer.size() < MIN_TUPLES_IN_CYCLE:
            episode_id += 1
            score = 0
            fucked_up_cnt = 0

            env = MeleeEnv(args.iso, pl_lst, fast_forward=True)
            env.start()
            now_s, _ = env.reset(enums.Stage.FINAL_DESTINATION)

            episode_memory1 = []
            episode_buffer1 = []
            episode_memory2 = []
            episode_buffer2 = []

            action_pair = [0, 0]

            r_sum1 = 0
            mask_sum1 = 1
            r_sum2 = 0
            mask_sum2 = 1

            last_state_idx1 = -1
            last_state_idx2 = -1

            pl_lst[0].ppo.actor_net.eval()
            pl_lst[1].ppo.actor_net.eval()
            for step_cnt in range(MAX_STEP):
                if step_cnt > 100:  # if step_cnt < 100, it's not started yet

                    now_action, act_data = pl_lst[0].act(now_s)
                    if act_data is not None:
                        episode_buffer1.append(
                            [now_s, act_data[0], act_data[1], step_cnt]
                        )
                        if args.print_fuck:
                            print(
                                "buffer index:",
                                len(episode_buffer1) - 1,
                                " / input action:",
                                act_data[0],
                            )
                    action_pair[0] = now_action
                    now_action, act_data = pl_lst[1].act(now_s)
                    if act_data is not None:
                        episode_buffer2.append(
                            [now_s, act_data[0], act_data[1], step_cnt]
                        )
                    action_pair[1] = now_action

                    now_s, r, done, _ = env.step(*action_pair)
                    mask = (1 - done) * 1
                    score += r[0]  # for log

                    r_sum1 += r[0]
                    mask_sum1 *= mask
                    r_sum2 += r[1]
                    mask_sum2 *= mask

                    if done:
                        # if finished, add last information to episode memory
                        temp = episode_buffer1[last_state_idx1]
                        episode_memory1.append(
                            [temp[0], temp[1], r_sum1, mask_sum1, temp[2]]
                        )
                        temp = episode_buffer2[last_state_idx2]
                        episode_memory2.append(
                            [temp[0], temp[1], r_sum1, mask_sum1, temp[2]]
                        )
                        break

                    if now_s[0].players[1].action_frame == 1:
                        # if agent's new action animation just started
                        p1_action = now_s[0].players[1].action
                        if p1_action in pl_lst[0].action_space.sensor:
                            # if agent's animation is in sensor set
                            # find action which caused agent's current animation
                            action_candidate = pl_lst[0].action_space.sensor[p1_action]
                            action_is_found = False
                            for i in range(
                                len(episode_buffer1) - 1, last_state_idx1, -1
                            ):

                                if episode_buffer1[i][3] > step_cnt - DELAY:
                                    # action can cause animation after 2 frames at least
                                    continue

                                if episode_buffer1[i][1] in action_candidate:
                                    if last_state_idx1 >= 0:
                                        # save last action and its consequence in episode memory
                                        temp = episode_buffer1[last_state_idx1]
                                        episode_memory1.append(
                                            [
                                                temp[0],
                                                temp[1],
                                                r_sum1,
                                                mask_sum1,
                                                temp[2],
                                            ]
                                        )
                                    r_sum1 = 0
                                    mask_sum1 = 1
                                    last_state_idx1 = i
                                    action_is_found = True
                                    break

                            if not action_is_found:
                                if args.print_fuck:
                                    print(
                                        "last_state_idx1:",
                                        last_state_idx1,
                                        " / animation:",
                                        p1_action,
                                    )
                                fucked_up_cnt += 1

                    if now_s[0].players[2].action_frame == 1:
                        # if agent's new action animation just started
                        p2_action = now_s[0].players[2].action
                        if p2_action in pl_lst[1].action_space.sensor:
                            # if agent's animation is in sensor set
                            # find action which caused agent's current animation
                            action_candidate = pl_lst[1].action_space.sensor[p2_action]
                            action_is_found = False
                            for i in range(
                                len(episode_buffer2) - 1, last_state_idx2, -1
                            ):

                                if episode_buffer2[i][3] > step_cnt - DELAY:
                                    # action can cause animation after 2 frames at least
                                    continue

                                if episode_buffer2[i][1] in action_candidate:
                                    if last_state_idx2 >= 0:
                                        # save last action and its consequence in episode memory
                                        temp = episode_buffer2[last_state_idx2]
                                        episode_memory2.append(
                                            [
                                                temp[0],
                                                temp[1],
                                                r_sum2,
                                                mask_sum2,
                                                temp[2],
                                            ]
                                        )
                                    r_sum2 = 0
                                    mask_sum2 = 1
                                    last_state_idx2 = i
                                    action_is_found = True
                                    break

                            if not action_is_found:
                                fucked_up_cnt += 1
                else:
                    action_pair = [0, 0]
                    now_s, _, _, _ = env.step(*action_pair)

            env.close()

            pl_lst[0].ppo.push_an_episode(episode_memory1, 1)
            pl_lst[0].ppo.push_an_episode(episode_memory2, 2)
            print(
                "episode:",
                episode_id,
                "\tbuffer length:",
                pl_lst[0].ppo.buffer.size(),
                "\tfucked up:",
                fucked_up_cnt,
            )

            with open(
                "log_" + args.env_name + ".csv", "a", encoding="utf-8"
            ) as outfile:
                outfile.write(str(episode_id) + "," + str(score) + "\n")
            scores.append(score)

        score_avg = np.mean(scores)
        print("cycle: ", cycle_id, "\tepisode: ", episode_id, "\tscore: ", score_avg)

        pl_lst[0].ppo.train()
        torch.save(pl_lst[0].ppo.actor_net, args.model_path + "actor_net.pt")
        torch.save(pl_lst[0].ppo.critic_net, args.model_path + "critic_net.pt")
        pl_lst[1].ppo.actor_net = torch.load(args.model_path + "actor_net.pt").to(
            device
        )
        pl_lst[1].ppo.critic_net = torch.load(args.model_path + "critic_net.pt").to(
            device
        )

    log_df = pd.read_csv("log_" + args.env_name + ".csv")
    plt.plot(log_df["episode_id"], log_df["score"])
    plt.show()


if __name__ == "__main__":
    run()
