# Q-Learn-for-chess.py
A code for a chess AI with RL (Reinforcement learning) in Python.

# Dependencies:
1. Python 3.10
2. Python chess module
3. Python numpy module

# How to train:
To train the RL model, one has to run the SARSA_Chess.py file. It will ask for three inputs, the amount of hours, 
minutes and seconds of training, all have to be integers. As a test run use 1 minute, with this one should see that the 
contents of the data files has changed. 
The code will return the amount of time it has been training and an exit code 0.

# How to play:
To play against the AI, one has to run the file named Playable_SARSA_chess.py, the moves have to be introduced as they 
are in these examples:
If you want to move the pawn from the e2 square to the e4 square you have to write: e2e4
If you want to castle king-side as white: e1g1
If you want to promote the g-file pawn as black: g7g8q (where the q means queen)
