#!/usr/bin/python3
import copy
from collections import deque

import melee

import os
import json
import melee as melee
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import pickle

import Args
from DataHandler_meleenv import *

from melee_env.agents.util import ObservationSpace, MyActionSpace

from PPO import Ppo
from parameters import (
    MAX_STEP,
    CYCLE_NUM,
    MIN_TUPLES_IN_CYCLE,
    STATE_DIM,
    ACTION_DIM,
    DELAY,
)
# for state_preprocessor
ppo = Ppo(STATE_DIM, ACTION_DIM, "cpu")

args = Args.get_args()

def load_data(replay_paths: str, player_character: melee.Character, opponent_character: melee.Character, itr):
    X_player = []
    Y_player = []

    X_opponent = []
    Y_opponent = []
    
    player_replay_id = []
    opponent_replay_id = []
    
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    
    episodes = []
    for idx, path in tqdm(enumerate(replay_paths)):
        observ = ObservationSpace()
        action_sp = MyActionSpace()
        console = melee.Console(system="file",
                                allow_old_version=True,
                                path=path)
        try:
            console.connect()
        except:
            console.stop()
            print('console failed to connect', path, time.time())
            continue

        gamestate: melee.GameState = console.step()
        player_port, opponent_port = get_ports(gamestate, player_character=player_character,
                                               opponent_character=opponent_character)
        if player_port == -1:
            print('bad port', path, gamestate.players.keys(), time.time())

            continue

        player: melee.PlayerState = gamestate.players.get(player_port)
        opponent: melee.PlayerState = gamestate.players.get(player_port)
        
        player_prev_game_last_idx = len(X_player)
        oppon_prev_game_last_idx = len(X_opponent)
        
        # episode_id += 1
        score = 0
        fucked_up_cnt = 0
        
        episode_memory = []
        episode_buffer = []

        action_pair = [0, 0]

        r_sum = 0
        mask_sum = 1

        last_state_idx = -1
        
        low_action_seq = [0, 0, 0, 0]
        gamestate_seq = [gamestate, gamestate, gamestate, gamestate]
        now_s = None # init gamestate
        act_data = [0, None]

        for step_cnt in range(MAX_STEP):
            if step_cnt > 100:        
                    
                # .act(s) -> (low level action idx, (high level action idx, action prob)))
                # we don't know the act prob
                now_action = get_low_action(player)
                
                low_action_seq.extend([now_action])
                del low_action_seq[0]
                now_action = low_action_seq[0]
                
                act_data[0] = low_seq2high(low_action_seq)
                if act_data is not None:
                    episode_buffer.extend(
                        [[now_s[0], act_data[0], act_data[1], step_cnt]] # 0 for act seq, 1 for prob
                    )    # we don't know action prob in offline setting -> None
                action_pair[0] = now_action
                action_pair[1] = get_low_action(opponent)
                    
                try:
                    gamestate: melee.GameState = console.step()
                except:
                    break
                
                gamestate_seq.extend([gamestate])
                del gamestate_seq[0]
                gamestate = gamestate_seq[0]
                
                if gamestate is None or gamestate.stage is None:
                    break
                
                player: melee.PlayerState = gamestate.players.get(player_port)
                opponent: melee.PlayerState = gamestate.players.get(opponent_port)
                if player is None or opponent is None:
                    break

                # now_s = (gamestate, prev_actions: np.array)
                now_s, r, done, _ = observ(gamestate, action_pair, player_port, opponent_port)
                mask = (1 - done) * 1
                score += r[0]  # for log

                r_sum += r[0]
                mask_sum *= mask
                
                if done:
                    # if finished, add last information to episode memory
                    temp = episode_buffer[last_state_idx]
                    episode_memory.extend(
                        [[temp[0], temp[1], r_sum, mask_sum, temp[2]]]
                    )
                    break

                if now_s[0].players[player_port].action_frame == 1:
                    # if agent's new action animation just started
                    p1_action = now_s[0].players[player_port].action
                    if p1_action in action_sp.sensor:
                        # if agent's animation is in sensor set
                        # find action which caused agent's current animation
                        action_candidate = action_sp.sensor[p1_action]
                        action_is_found = False
                        for i in range(len(episode_buffer) - 1, last_state_idx, -1):

                            if episode_buffer[i][3] > step_cnt - 2:#DELAY:
                                # action can cause animation after 2 frames at least
                                continue

                            if episode_buffer[i][1] in action_candidate:
                                if last_state_idx >= 0:
                                    # save last action and its consequence in episode memory
                                    temp = episode_buffer[last_state_idx]
                                    # episode_memory.append(
                                    #     [temp[0], temp[1], r_sum, mask_sum, temp[2]]
                                    # )
                                
                                    observations.extend([ppo.state_preprocessor((temp[0], None), player_port, opponent_port)[0]])
                                    actions.extend([temp[1]])
                                    rewards.extend([r_sum])
                                    terminals.extend([mask_sum])
                                    timeouts.extend([False])
                                
                                r_sum = 0
                                mask_sum = 1
                                last_state_idx = i
                                action_is_found = True
                                break

                        if not action_is_found:
                            fucked_up_cnt += 1
                
            else:
                action_pair = [0, 0]
                gamestate: melee.GameState = console.step()
                gamestate_seq.extend([gamestate])
                del gamestate_seq[0]
                gamestate = gamestate_seq[0]
                
                now_s, _, _, _ = observ(gamestate, action_pair, player_port, opponent_port)
                
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    timeouts = np.array(timeouts)
    
    replay_data = {"observations": observations, "actions": actions, "rewards": rewards, "terminals":terminals, "timeouts":timeouts}
    return replay_data

def process_replays(replays: dict, c1: melee.Character, c2: melee.Character, s: melee.Stage, iteration : int):
    #print(f'Data/{c1.name}_{c2.name}_on_{s.name}_data.pkl')
    #print(f'Data/{c2.name}_{c1.name}_on_{s.name}_data.pkl')

    replay_paths = replays[f'{c1.name}_{c2.name}'][s.name]

    replay_data = load_data(replay_paths, c1, c2, iteration)

    data_file_path = f'./Data/{c1.name}_{c2.name}_on_{s.name}_data.pkl' # 수정 필
    with open(data_file_path, 'wb') as file:
        pickle.dump(replay_data, file)

    
if __name__ == '__main__':
    # Mass Generate
    f = open('replays.json', 'r')
    replays = json.load(f)
    characters = [melee.Character.FOX, melee.Character.JIGGLYPUFF, melee.Character.MARTH, melee.Character.CPTFALCON, melee.Character.FALCO]
    #characters = list(melee.Character)
    total_n = 0
    succed_n = 0
    exceptions = []
    iteration = 0
    for e, c1 in enumerate(characters):
        for c2 in characters:
            for s in [melee.Stage.FINAL_DESTINATION]:
                # try:
                if (os.path.exists(f'./Data/{c1.name}_{c2.name}_on_{s.name}_data.pkl')):
                    iteration += 1
                    continue
                process_replays(replays, c1, c2, s, iteration)
                total_n += 1
                succed_n += 1
                # except Exception as exc:
                #     if type(exc) is KeyError:
                #         total_n += 1
                #         exceptions.append(exc)
                #     else:
                #         raise exc
                iteration += 1
            break
        break

                        
    
    print(f"Ratio: {succed_n}/{total_n}")
    print(exceptions)
