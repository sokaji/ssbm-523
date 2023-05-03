# SSBM Imitation Learning 

A mix of imitation learning techniques for training agents in Super Smash Brothers Melee. A quick and dirty implementation of different networks for the purpose of testing different methodologies in network design, ranging from different activation functions, loss functions, normalization, clipping, optimizers, etc. Also tested different architectures, ranging from fully connected dense networks to utilizing a basic GPT like transformer model. The code is not fully parameterized and requires some changes to "hard coded" values. 

Uses the libmelee API to read in raw data values from replays and game state, which hooks into the Slippi platform for SSBM emulation. 

Note: Modified the implementation from https://github.com/AI-Spawn/ssbm-bot

## Important files: 

**train.py:** Basic training file, similar to the source. 

**train_new.py** Tested different architectures, epochs, and other network parameters. 

**train_transfomer.py** Used multi-head self attention for the use in a GPT type transformer model for imitation learning.  

**TransformerBot.py** The bot being controlled needed to be changed to accept the transformer model. 

**TransformerDuel.py** Same with the bot, the dueling script needed to change to handle the transformer model. 

**General Changes:** 
- Included use of multiple epochs
- Added plotting of loss/accuracy
- Modified model saving to be consistent with Tensorflow
- Improved data loading


## Usage (Adapted from ssbm-bot)

**Step 0:** Follow the [libmelee](https://github.com/altf4/libmelee) setup instruction. Set the dolphin path in `Args.py`

**Step 1**: [Get a melee ISO](https://dolphin-emu.org/docs/guides/ripping-games/), name it `SSBM.iso` and put it in the main folder.

**Step 2:** Location for a sizeable dataset [here](https://drive.google.com/file/d/1ab6ovA46tfiPZ2Y3a_yS1J3k3656yQ8f/edit) This is maintained by the creator of libmelee and contains GBs of files at the time of writing. 

**Step 3:** Change the `replay_folder` variable to the path to your dataset, and run `organize_replays.py`

**Step 4:** Run `generate_data.py` . Depending on the size of your dataset, this may take a very long time.

**Step 5:**  Set  `player_character`, `opponent_character`, and `stage` to your desired targets and run `train.py`. You will need to tune the optimizer, learning rate, network structure with different targets. 

Note: If you are using any of the modified files, replace `train.py` with the other training files mentioned above. 

**Step 6:** Set the same targets in `duel.py` and run it. You can very the models "attack weighting" by changing the denominator in `Datahandler.py` line 231

Note: Make sure to change to the transformer specific files if testing transformer capabilities. 




















