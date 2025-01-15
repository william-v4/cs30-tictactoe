import customtkinter as gui
from tkinter import *
import warnings
import random

# times to train AI
EPISODES = 1000
# disable output for faster training
fasttrain = False


# suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="CTkLabel Warning: Given image is not CTkImage")
warnings.filterwarnings("ignore", category=UserWarning, message="CTkButton Warning: Given image is not CTkImage")
# load fonts
gui.FontManager.load_font("Roboto-Regular.ttf")
# set text styles
titletext = ("Roboto", 24)
normaltext = ("Roboto", 16)
# gui window
window = gui.CTk()
# window title
window.title("Tic Tac Toe")
# game mode
gamemode = StringVar(window)
# default is singleplayer
gamemode.set("singleplayer")
# Q-values table
memory = {}
# game board
board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
# if AI is training
training = False


# clear the window
def clear():
    # for every widget in the window
    for widget in window.winfo_children():
        # eject them
        widget.destroy()

def gameconfig(arg=None):
    # global variables
    global gamemode
    print(gamemode.get())
    # clear window
    clear()
    # title
    gui.CTkLabel(window, text="Welcome to Tic Tac Toe", font=titletext).grid(row=1, column=1, columnspan=2, padx=10, pady=10)
    # dropdown for game mode (reruns this function when changed to show new settings)
    gui.CTkOptionMenu(window, values=["singleplayer", "multiplayer"], variable=gamemode, font=normaltext, command=gameconfig).grid(row=2, column=1, columnspan=2, padx=10, pady=10)
    # button to start the game
    gui.CTkButton(window, text="", command=startgame, image=PhotoImage(file="start.png")).grid(row=4, column=1, columnspan=2, padx=10, pady=10)

# clear the board
def resetboard():
    global board
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

# start the game
def startgame():
    # global variables
    global turn, slots, memory
    # clear window
    clear()
    # clear board
    resetboard()
    # create the displayed board
    slots = {}
    createslots()
    # if singleplayer
    if gamemode.get() == "singleplayer":
        # if there's nothing in memory, train the AI
        if memory == {}:
            print("training AI")
            train(EPISODES)
            print("training done")
        # randomize who goes first
        turn = random.choice(["player", "ai"])
        # if AI goes first
        if turn == "ai":
            # AI makes a move
            AImove()
    # if multiplayer
    elif gamemode.get() == "multiplayer":
        # randomize who goes first
        turn = random.choice(["x", "o"])

# create buttons for each grid on Tic Tac Toe board
def createslots():
    global slots
    # for every grid in 3x3
    for i in range (3):
        for j in range (3):
            # create and place button that calls move(i, j) when clicked
            slots[i, j] = gui.CTkButton(window, text="", command=lambda i=i, j=j: move(i, j), image=PhotoImage(file="empty.png"))
            slots[i, j].grid(row=i+1, column=j+1, padx=5, pady=5)

# train the AI
def train(times : int, rate : float = 0.5, longterm : float = 0.9, randomthreshold : float = 0.0):
    global memory, turn, training, reward, board, done
    training = True
    # let AI use x
    turn = "x"
    # for number of times specified in parameter
    for episode in range(times):
        if not fasttrain:
            print("episode", episode)
        # reset the board
        resetboard()
        # continue playing round until it is over
        done = False
        while not done:
            # chance (based on randomthreshold) to make a random move for exploration
            if random.random() < randomthreshold:
                # pick a random move
                action = random.choice(availablemoves())
            # otherwise, pick the best move for current state
            else:
                # get all possible Q-values for all possible moves
                qvalues = [getqvalue(getstate(), a) for a in availablemoves()]
                # get the highest q-value (associated with the best move)
                maxq = max(qvalues)
                # get all moves that have that value
                bestmoves = []
                for x in availablemoves():
                    if getqvalue(getstate(), x) == maxq:
                        bestmoves.append(x)
                # pick a random move from the best moves
                action = random.choice(bestmoves)
            # make the move (tuple format, so needs to be unpacked)
            move(*action)
            # ensure a default if the state and action are not in memory
            if (getstate(), action) not in memory:
                memory[(getstate(), action)] = 0.0
            # if game is over
            if done:
                # store the q-value for the state and action (bellman equation)
                memory[(getstate(), action)] = memory[(getstate(), action)] + rate * (reward - memory[(getstate(), action)])
                # reset the board
                resetboard()
            # otherwise, calculate the q-value for the next state and action
            else:
                # get all possible next moves
                nextactions = availablemoves()
                # get the actions with the highest q-value from memory. If it doesn't exist, default to 0.0. If there are no next actions, default to 0. 
                maxqnext = max([memory.get((getstate(), a), 0.0) for a in nextactions]) if nextactions else 0
                # store the q-value for the state and action (bellman equation) but include next state's q-value for long-term reward
                memory[(getstate(), action)] = memory[(getstate(), action)] + rate * (reward + longterm * maxqnext - memory[(getstate(), action)])
    # training is done
    training = False
    # reset UI for player
    resetboard()
    clear()
    createslots()


# get available moves
def availablemoves() -> list:
    global board
    # for every i and j index between 0 and 2, if the slot is empty (0), append to list. return list
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]

# get board state (storable in list)
def getstate() -> tuple:
    global board
    return tuple(tuple(row) for row in board)

# get q-value for a state and action
def getqvalue(state : tuple, action : tuple) -> float:
    global memory
    # fetch the q-value from memory. If it doesn't exist, default to 0.0
    return memory.get((state, action), 0.0)

# make a move
def move(i : int, j : int):
    global board, turn, reward
    # if the slot is not empty, return False
    if board[i][j] != 0:
        return False
    # set slot on the board based on player
    match turn:
        case "player":
            board[i][j] = 1
        case "ai":
            board[i][j] = -1
        case "x":
            board[i][j] = 1
        case "o":
            board[i][j] = -1
    # disable corresponding slot
    slots[i, j].configure(state=DISABLED)
    # update the displayed board if not training
    if fasttrain and training:
        pass
    else:
        updatedisplay()
    # check for win
    checkwin()
    # flip the turns
    if turn == "player":
        turn = "ai"
        AImove()
    elif turn == "ai":
        turn = "player"
    elif turn == "x":
        turn = "o"
    elif turn == "o":
        turn = "x"

# update the displayed board
def updatedisplay():
    global board, slots
    # image paths
    imgs = {1: "x.png", -1: "o.png", 0: "empty.png"}
    # for every slot in 3x3 grid
    for i in range(3):
        for j in range(3):
            # update button to show corresponding image
            slots[i, j].configure(image=PhotoImage(file=imgs[board[i][j]]))
    # refresh
    window.update()

# check if someone won
def checkwin() -> int:
    global reward, winner, board, done
    # for every row and column
    for i in range(3):
        # if the sum of the row or column is 3 or -3
        if abs(sum(board[i])) == 3 or abs(sum(row[i] for row in board)) == 3:
            # the current player wins
            winner = turn
            # reward for AI is 2 during training
            reward = 2
            # training episode done
            done = True
            # do not show game over screen during training
            if not training:
                gameover()
            return 1
    # check diagonals
    if abs(board[0][0] + board[1][1] + board[2][2]) == 3 or abs(board[0][2] + board[1][1] + board[2][0]) == 3:
        # current player wins
        winner = turn
        # reward for AI is 2 during training
        reward = 2
        # training episode done
        done = True
        # do not show game over screen during training
        if not training:
            gameover()
        return 1
    # check if the board is full (draw)
    if all(all(slot != 0 for slot in row) for row in board):
        # nobody wins
        winner = "nobody"
        # reward for AI is 1 during training
        reward = 1
        # training episode done
        done = True
        # do not show game over screen during training
        if not training:
            gameover()
        return 2
    # if no win or draw, reward is 0 and return 0
    reward = 0
    return 0

# game over
def gameover():
    # global variables
    global winner
    # clear window
    clear()
    # display winner
    gui.CTkLabel(window, text=f"{winner} wins!", font=titletext).grid(row=1, column=1, padx=10, pady=10)
    # button to play again
    gui.CTkButton(window, text="", command=startgame, image=PhotoImage(file="restart.png")).grid(row=2, column=1, padx=10, pady=10)

# predict the next move
def AImove():
    # global variables
    global turn, memory, board, winner, slots
    # get all possible q-values for all possible moves
    qvalues = [getqvalue(getstate(), a) for a in availablemoves()]
    # get the highest q-value (associated with the best move)
    maxq = max(qvalues)
    # get all moves that have that value
    bestmoves = []
    for x in availablemoves():
        if getqvalue(getstate(), x) == maxq:
            bestmoves.append(x)
    # pick a random move from the best moves
    action = random.choice(bestmoves)
    # if there is no best move, pick a random move
    if not bestmoves:
        action = random.choice(availablemoves())
    # make the move
    move(*action)


"""
MAIN THREAD
"""
# ask for initial game parameters
gameconfig()
# start window
window.mainloop()