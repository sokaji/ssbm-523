import Args
import GameManager
import melee
import platform

import os
import torch
import trainer
import numpy as np

args = Args.get_args()



if __name__ == '__main__':
    character = melee.Character.MARTH
    opponent = melee.Character.CPTFALCON if not args.compete else character
    stage = melee.Stage.FINAL_DESTINATION

    model = torch.load(f"models/{character.name}_v_{opponent.name}_on_{stage.name}")
    # print(loaded)
    # model = network.Network(19*2, 10)
    # model.load_state_dict(loaded)
    game = GameManager.Game(args)
    game.enterMatch(cpu_level=9, opponant_character=opponent,
                    player_character=character,
                    stage=stage, rules=False)



    num_buttons = len(trainer.buttons) + 1
    axis_size = 3
    num_c = 5
    max = []
    with torch.no_grad():
        while True:
            gamestate = game.get_gamestate()

            inp = trainer.generate_input(gamestate, 1, 2)
            out = model(torch.Tensor(inp)).detach().numpy()




            action = np.argmax(out)
            decoded =
            # print(action)
            # print(button)
            # print(move)
            # print('----------')
            # print(trainer.buttons)
            move_x = int(move / 3 + 0.5 / 3)
            move_y = move - 3 * move_x
            if button > 0:
                game.controller.press_button(trainer.buttons[button - 1][0])

            game.controller.tilt_analog(melee.Button.BUTTON_MAIN, move_x / 2, move_y / 2)
            gamestate = game.get_gamestate()
            game.controller.release_all()
