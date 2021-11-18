import melee


def clamp(n, smallest, largest): return max(smallest, min(n, largest))


attacking_list = [melee.Action.BACKWARD_TECH, melee.Action.BAT_SWING_1, melee.Action.BAT_SWING_2,
                  melee.Action.BAT_SWING_3, melee.Action.BAT_SWING_4, melee.Action.BEAM_SWORD_SWING_1,
                  melee.Action.BEAM_SWORD_SWING_2, melee.Action.BEAM_SWORD_SWING_3,
                  melee.Action.BEAM_SWORD_SWING_4, melee.Action.CEILING_TECH, melee.Action.DAMAGE_ICE,
                  melee.Action.DAMAGE_BIND, melee.Action.DAMAGE_SONG, melee.Action.DAMAGE_SCREW,
                  melee.Action.DAMAGE_AIR_1, melee.Action.DAMAGE_AIR_2, melee.Action.DAMAGE_AIR_3,
                  melee.Action.DAMAGE_FLY_HIGH, melee.Action.DAMAGE_FLY_LOW, melee.Action.DAMAGE_FLY_NEUTRAL,
                  melee.Action.DAMAGE_FLY_ROLL, melee.Action.DAMAGE_FLY_TOP, melee.Action.DAMAGE_GROUND,
                  melee.Action.DAMAGE_HIGH_1, melee.Action.DAMAGE_HIGH_2, melee.Action.DAMAGE_HIGH_3,
                  melee.Action.DAMAGE_ICE_JUMP, melee.Action.DAMAGE_LOW_1, melee.Action.DAMAGE_LOW_2,
                  melee.Action.DAMAGE_LOW_3, melee.Action.DAMAGE_NEUTRAL_1, melee.Action.DAMAGE_NEUTRAL_2,
                  melee.Action.DAMAGE_NEUTRAL_3, melee.Action.DAMAGE_SCREW, melee.Action.DAMAGE_SCREW_AIR,
                  melee.Action.DAMAGE_SONG, melee.Action.DAMAGE_SONG_RV, melee.Action.DAMAGE_SONG_WAIT,
                  melee.Action.DASH_ATTACK, melee.Action.DK_GROUND_POUND, melee.Action.DK_GROUND_POUND_START,
                  melee.Action.DOWNSMASH, melee.Action.DOWNTILT, melee.Action.DOWN_B_GROUND_START,
                  melee.Action.DOWN_B_GROUND, melee.Action.DOWN_B_STUN, melee.Action.DOWN_B_AIR,
                  melee.Action.EDGE_GETUP_QUICK, melee.Action.EDGE_ATTACK_SLOW, melee.Action.FIREFOX_GROUND,
                  melee.Action.FIREFOX_AIR, melee.Action.FIREFOX_WAIT_AIR, melee.Action.FIREFOX_WAIT_GROUND,
                  melee.Action.FIRE_FLOWER_SHOOT, melee.Action.FIRE_FLOWER_SHOOT_AIR, melee.Action.FORWARD_TECH,
                  melee.Action.FOX_ILLUSION, melee.Action.FOX_ILLUSION_START,
                  melee.Action.FOX_ILLUSION_SHORTENED, melee.Action.FSMASH_HIGH, melee.Action.FSMASH_LOW,
                  melee.Action.FSMASH_MID, melee.Action.FSMASH_MID_HIGH, melee.Action.FSMASH_MID_LOW,
                  melee.Action.FTILT_HIGH, melee.Action.FTILT_LOW, melee.Action.FTILT_MID,
                  melee.Action.FTILT_LOW_MID, melee.Action.FTILT_HIGH_MID, melee.Action.GETUP_ATTACK,
                  melee.Action.GUN_SHOOT_AIR, melee.Action.GUN_SHOOT, melee.Action.JUMPING_ARIAL_BACKWARD,
                  melee.Action.JUMPING_ARIAL_FORWARD, melee.Action.NEUTRAL_ATTACK_1,
                  melee.Action.NEUTRAL_ATTACK_2, melee.Action.NEUTRAL_ATTACK_3,
                  melee.Action.NEUTRAL_B_ATTACKING, melee.Action.NEUTRAL_B_ATTACKING_AIR,
                  melee.Action.PARASOL_SWING_1, melee.Action.PARASOL_SWING_2, melee.Action.PARASOL_SWING_3,
                  melee.Action.PARASOL_SWING_4, melee.Action.SWORD_DANCE_1,
                  melee.Action.UAIR, melee.Action.UAIR_LANDING,
                  melee.Action.UPSMASH, melee.Action.UPTILT, melee.Action.UP_B_AIR, melee.Action.UP_B_GROUND,
                  melee.Action.SWORD_DANCE_1_AIR,
                  melee.Action.SWORD_DANCE_2_HIGH, melee.Action.SWORD_DANCE_2_HIGH_AIR,
                  melee.Action.SWORD_DANCE_2_MID, melee.Action.SWORD_DANCE_2_MID_AIR, melee.Action.SWORD_DANCE_2_HIGH,
                  melee.Action.SWORD_DANCE_2_HIGH_AIR, melee.Action.SWORD_DANCE_3_HIGH,
                  melee.Action.SWORD_DANCE_3_HIGH_AIR, melee.Action.SWORD_DANCE_3_LOW,
                  melee.Action.SWORD_DANCE_3_LOW_AIR, melee.Action.SWORD_DANCE_3_MID,
                  melee.Action.SWORD_DANCE_3_MID_AIR, melee.Action.SWORD_DANCE_4_HIGH,
                  melee.Action.SWORD_DANCE_4_HIGH_AIR, melee.Action.SWORD_DANCE_4_LOW, melee.Action.SWORD_DANCE_4_LOW,
                  melee.Action.SWORD_DANCE_4_LOW_AIR, melee.Action.SWORD_DANCE_4_MID,
                  melee.Action.SWORD_DANCE_4_MID_AIR]

dead_list = [melee.Action.DEAD_FALL, melee.Action.DEAD_DOWN, melee.Action.DEAD_FLY, melee.Action.DEAD_FLY_SPLATTER,
             melee.Action.DEAD_FLY_SPLATTER_FLAT, melee.Action.DEAD_FLY_SPLATTER_FLAT_ICE,
             melee.Action.DEAD_FLY_SPLATTER_ICE, melee.Action.DEAD_FLY_STAR, melee.Action.DEAD_FLY_STAR_ICE,
             melee.Action.DEAD_LEFT, melee.Action.DEAD_RIGHT, melee.Action.DEAD_UP, melee.Action.ON_HALO_DESCENT]