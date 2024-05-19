from melee import enums
from melee.enums import Action
from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import NOOP, CPU
import argparse
from collections import deque
import random

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument(
    "--iso",
    default="../ssbm.iso",
    type=str,
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO",
)

args = parser.parse_args()

players = [NOOP(enums.Character.FOX), CPU(enums.Character.FOX, 8)]

env = MeleeEnv(args.iso, players, fast_forward=True)
env.start()

recorded = [
    Action.DASHING,
    Action.TURNING,
    Action.JUMPING_FORWARD,
    Action.JUMPING_ARIAL_FORWARD,
    Action.JUMPING_BACKWARD,
    Action.JUMPING_ARIAL_BACKWARD,
    Action.NEUTRAL_ATTACK_1,
    Action.NEUTRAL_ATTACK_2,
    Action.LOOPING_ATTACK_START,
    Action.LOOPING_ATTACK_MIDDLE,
    Action.LOOPING_ATTACK_END,
    Action.NAIR,
    Action.NAIR_LANDING,
    Action.DASH_ATTACK,
    Action.FSMASH_MID,
    Action.FAIR,
    Action.FAIR_LANDING,
    Action.BAIR,
    Action.BAIR_LANDING,
    Action.UPSMASH,
    Action.UAIR,
    Action.UAIR_LANDING,
    Action.DOWNSMASH,
    Action.DAIR,
    Action.DAIR_LANDING,
    Action.FTILT_MID,
    Action.UPTILT,
    Action.DOWNTILT,
    Action.CROUCH_END,
    Action.LASER_GUN_PULL,
    Action.NEUTRAL_B_CHARGING,
    Action.NEUTRAL_B_ATTACKING,
    Action.NEUTRAL_B_FULL_CHARGE,
    Action.WAIT_ITEM,
    Action.NEUTRAL_B_CHARGING_AIR,
    Action.NEUTRAL_B_ATTACKING_AIR,
    Action.NEUTRAL_B_FULL_CHARGE_AIR,
    Action.SWORD_DANCE_1,
    Action.SWORD_DANCE_2_HIGH,
    Action.SWORD_DANCE_2_MID,
    Action.SWORD_DANCE_3_HIGH,
    Action.LANDING_SPECIAL,
    Action.SWORD_DANCE_1_AIR,
    Action.SWORD_DANCE_2_HIGH_AIR,
    Action.SWORD_DANCE_3_LOW,
    Action.SWORD_DANCE_3_MID,
    Action.SWORD_DANCE_3_LOW_AIR,
    Action.SWORD_DANCE_3_MID_AIR,
    Action.SWORD_DANCE_3_HIGH_AIR,
    Action.SWORD_DANCE_4_LOW,
    Action.SWORD_DANCE_4_MID,
    Action.SWORD_DANCE_4_HIGH,
    Action.DOWN_B_GROUND_START,
    Action.DOWN_B_GROUND,
    Action.DOWN_B_STUN,
    Action.DOWN_B_AIR,
    Action.SHINE_RELEASE_AIR,
    Action.GRAB,
    Action.GRAB_PULLING,
    Action.GRAB_WAIT,
    Action.GRAB_BREAK,
    Action.GRAB_RUNNING,
    Action.GRAB_PUMMEL,
    Action.THROW_FORWARD,
    Action.THROW_BACK,
    Action.THROW_UP,
    Action.THROW_DOWN,
    Action.SHIELD_START,
    Action.SHIELD_STUN,
    Action.SHIELD_RELEASE,
    Action.ROLL_FORWARD,
    Action.ROLL_BACKWARD,
    Action.SPOTDODGE,
    Action.EDGE_JUMP_1_QUICK,
    Action.EDGE_JUMP_2_QUICK,
    Action.EDGE_JUMP_1_SLOW,
    Action.EDGE_JUMP_2_SLOW,
    Action.EDGE_ATTACK_QUICK,
    Action.EDGE_ATTACK_SLOW,
    Action.EDGE_GETUP_QUICK,
    Action.EDGE_GETUP_SLOW,
    Action.EDGE_ROLL_QUICK,
    Action.EDGE_ROLL_SLOW,
    Action.GETUP_ATTACK,
    Action.NEUTRAL_GETUP,
    Action.GROUND_ROLL_BACKWARD_DOWN,
    Action.GROUND_ROLL_FORWARD_DOWN,

    Action.GRAB_PULL,  # 조사 필요
    Action.MARTH_COUNTER,
    Action.SHINE_TURN,
    Action.FTILT_HIGH,

    Action.STANDING,
    Action.KNEE_BEND,
    Action.ENTRY,
    Action.ENTRY_START,
    Action.ENTRY_END,
    Action.FALLING,
    Action.LANDING,
    Action.DEAD_FALL,
    Action.DEAD_DOWN,
    Action.ON_HALO_DESCENT,
    Action.GRABBED,
    Action.GRAB_PUMMELED,
    Action.CROUCH_START,
    Action.TECH_MISS_UP,
    Action.TECH_MISS_DOWN,
    Action.WALK_SLOW,
    Action.CROUCHING,
    Action.THROWN_BACK,
    Action.THROWN_COPY_STAR,
    Action.THROWN_CRAZY_HAND,
    Action.THROWN_DOWN,
    Action.THROWN_DOWN_2,
    Action.THROWN_FB,
    Action.THROWN_FF,
    Action.THROWN_FORWARD,
    Action.THROWN_F_HIGH,
    Action.THROWN_F_LOW,
    Action.THROWN_UP,
    Action.DAMAGE_AIR_1,
    Action.DAMAGE_AIR_2,
    Action.DAMAGE_AIR_3,
    Action.DAMAGE_BIND,
    Action.DAMAGE_FLY_HIGH,
    Action.DAMAGE_FLY_LOW,
    Action.DAMAGE_FLY_NEUTRAL,
    Action.DAMAGE_FLY_ROLL,
    Action.DAMAGE_FLY_TOP,
    Action.DAMAGE_GROUND,
    Action.DAMAGE_HIGH_1,
    Action.DAMAGE_HIGH_2,
    Action.DAMAGE_HIGH_3,
    Action.DAMAGE_ICE,
    Action.DAMAGE_ICE_JUMP,
    Action.DAMAGE_LOW_1,
    Action.DAMAGE_LOW_2,
    Action.DAMAGE_LOW_3,
    Action.DAMAGE_NEUTRAL_1,
    Action.DAMAGE_NEUTRAL_2,
    Action.DAMAGE_NEUTRAL_3,
    Action.DAMAGE_SCREW,
    Action.DAMAGE_SCREW_AIR,
    Action.DAMAGE_SONG,
    Action.DAMAGE_SONG_RV,
    Action.DAMAGE_SONG_WAIT,
    Action.UNKNOWN_ANIMATION,
    Action.EDGE_ATTACK_QUICK,
    Action.EDGE_ATTACK_SLOW,
    Action.EDGE_CATCHING,
    Action.EDGE_GETUP_QUICK,
    Action.EDGE_GETUP_SLOW,
    Action.EDGE_HANGING,
    Action.EDGE_JUMP_1_QUICK,
    Action.EDGE_JUMP_1_SLOW,
    Action.EDGE_JUMP_2_QUICK,
    Action.EDGE_JUMP_2_SLOW,
    Action.EDGE_ROLL_QUICK,
    Action.EDGE_ROLL_SLOW,
    Action.EDGE_TEETERING,
    Action.EDGE_TEETERING_START,
]

now_obs, _ = env.reset(enums.Stage.BATTLEFIELD)
for step_cnt in range(3600):
    if step_cnt > 120:

        action_pair = [random.randint(0, 44), 0]

        next_obs, r, done, _ = env.step(*action_pair)
    else:
        action_pair = [0, 0]
        next_obs, r, done, _ = env.step(*action_pair)

    if next_obs[0].players[1].action not in recorded:
        print("new action state: ", next_obs[0].players[1].action)

env.close()