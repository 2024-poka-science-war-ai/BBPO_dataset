import numpy as np
import melee

import MovesList
import melee_env
from melee_env.agents.util import MyActionSpace

import torch
import math

framedata = melee.FrameData()

low_analog = 0.2
high_analog = 0.8


def controller_states_different(new_player: melee.PlayerState, old_player: melee.PlayerState):
    new: melee.ControllerState = new_player.controller_state
    old: melee.ControllerState = old_player.controller_state

    # if generate_output(new_player) == generate_output(old_player):
    #     return False

    for btns in MovesList.buttons:
        # for b in melee.enums.Button:
        for b in btns:
            if new.button.get(b) != old.button.get(b) and new.button.get(b):
                return True

    if new.c_stick[0] < low_analog and old.c_stick[0] >= low_analog:
        return True

    if new.c_stick[0] > high_analog and old.c_stick[0] <= high_analog:
        return True

    if new.c_stick[1] < low_analog and old.c_stick[1] >= low_analog:
        return True

    if new.c_stick[1] > high_analog and old.c_stick[1] <= high_analog:
        return True

    if new.main_stick[0] < low_analog and old.main_stick[0] >= low_analog:
        return True

    if new.main_stick[0] > high_analog and old.main_stick[0] <= high_analog:
        return True

    if new.main_stick[1] < low_analog and old.main_stick[1] >= low_analog:
        return True

    if new.main_stick[1] > high_analog and old.main_stick[1] <= high_analog:
        return True

    return False

    # return generate_output(new) != generate_output(old)


def get_ports(gamestate: melee.GameState, player_character: melee.Character, opponent_character: melee.Character):
    if gamestate is None:
        return -1, -1
    ports = list(gamestate.players.keys())
    if len(ports) != 2:
        return -1, -1
    player_port = ports[0]
    opponent_port = ports[1]
    p1: melee.PlayerState = gamestate.players.get(player_port)
    p2: melee.PlayerState = gamestate.players.get(opponent_port)

    if p1.character == player_character and p2.character == opponent_character:
        player_port = ports[0]
        opponent_port = ports[1]
    elif p2.character == player_character and p1.character == opponent_character:
        player_port = ports[1]
        opponent_port = ports[0]
    else:
        print(p1.character, p2.character)
        player_port = -1
        opponent_port = -1
    return player_port, opponent_port


def get_player_obs(player: melee.PlayerState) -> list:
    x = player.position.x / 100
    y = player.position.y / 50
    shield = player.shield_strength / 60
    off_stage = 1 if player.off_stage else 0

    percent = player.percent / 100
    is_attacking = 1 if framedata.is_attack(player.character, player.action) else 0
    on_ground = 1 if player.on_ground else 0
    
    status = float(player.action.value)

    facing = 1 if player.facing else -1
    
    in_hitstun = 1 if player.hitlag_left else 0
    is_invulnerable = 1 if player.invulnerable else 0
    
    jumps_left = player.jumps_left

    attack_state = framedata.attack_state(player.character, player.action, player.action_frame)
    attack_active = 1 if attack_state == melee.AttackState.ATTACKING else 0
    attack_cooldown = 1 if attack_state == melee.AttackState.COOLDOWN else 0
    attack_windup = 1 if attack_state == melee.AttackState.WINDUP else 0

    is_bmove = 1 if framedata.is_bmove(player.character, player.action) else 0

    stock = player.stock
    return [
        shield, on_ground, is_attacking,
        off_stage,
        x, y,
        percent,
        facing,
        in_hitstun,
        is_invulnerable,
        jumps_left,
        status,
        attack_active,
        attack_cooldown,
        attack_windup
    ]


def generate_input( player: melee.PlayerState, opponent: melee.PlayerState):
    direction = 1 if player.position.x < opponent.position.x else -1
    
    obs = [
        (player.position.x - opponent.position.x) / 20, (player.position.y - opponent.position.y) / 10,
        direction,
        1 if player.position.x > opponent.position.x else -1,
        1 if player.position.y > opponent.position.y else -1,
        math.sqrt(pow(player.position.x - opponent.position.x, 2) + pow(player.position.y - opponent.position.y, 2))
    ]
    obs += get_player_obs(player)
    obs += get_player_obs(opponent)

    return np.array(obs).flatten()


def get_low_action(player: melee.PlayerState):
    from melee import enums
    controller_state = player.controller_state
    x, y = controller_state.main_stick
    button_a = controller_state.button[enums.Button.BUTTON_A]
    button_b = controller_state.button[enums.Button.BUTTON_B] and not button_a
    button_z = controller_state.button[enums.Button.BUTTON_Z] and not button_b and not button_a
    button_r = controller_state.button[enums.Button.BUTTON_R] and not button_z and not button_b and not button_a


    # 방향 계산
    angle = math.atan2(y - 0.5, x - 0.5)
    mag = math.hypot(x - 0.5, y - 0.5)
    
    # 버튼 상태에 따른 base value 설정
    if button_a:
        if 0.1 <= mag < 0.5:
            base = 13
        else:
            base = 9
        
    elif button_b:
        base = 18
    elif button_z:
        base = 23
        return base
    elif button_r:
        base = 24
    else:
        base = 0

    if base == 0:
        if mag < 0.1:  # 중립
            return base
        elif -math.pi/8 <= angle < math.pi/8:
            base += 1  # Right
        elif math.pi/8 <= angle < 3*math.pi/8:
            base += 5  # Up/Right with button
        elif 3*math.pi/8 <= angle < 5*math.pi/8:
            base += 3  # Up
        elif 5*math.pi/8 <= angle < 7*math.pi/8:
            base += 6  # Up/Left with button
        elif -3*math.pi/8 <= angle < -math.pi/8:
            base += 7  # Down/Right with button
        elif -5*math.pi/8 <= angle < -3*math.pi/8:
            base += 4  # Down
        elif -7*math.pi/8 <= angle < -5*math.pi/8:
            base += 8  # Down/Left with button
        else:
            base += 2  # Left
        # print(x, y, base)

    else:
        if mag < 0.1:  # 중립
            return base
        elif -math.pi/4 <= angle < math.pi/4:
            base += 1  # Right
        elif math.pi/4 <= angle < 3*math.pi/4:
            base += 3  # Up
        elif -3*math.pi/4 <= angle < -math.pi/4:
            base += 4  # Down
            if button_r:
                base -= 1
        else:
            base += 2  # Left
    
    return base

def low_seq2high(low_seq):
    action_sp = MyActionSpace()

    min_dist = 2**6
    
    ans = 0
    for i, high_seq in enumerate(action_sp.high_action_space):
        temp = action_dist(low_seq, high_seq.copy())
        if min_dist > temp:
            min_dist = temp
            ans = i
            
    return ans
            
            
    
def action_dist(low:list, dest:list):
    if len(dest) == 2:
        dest.extend([0, 0])
    dist = 0
    
    for i in range(len(dest)):
        if dest[i] != low[i]:
            dist += 2**i
    
    return dist

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])