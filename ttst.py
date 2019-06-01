import argparse
import itertools
import logging
import numpy as np
import os
import pickle
import time

# Learning parameters
START_WEIGHT = 6
PENALTY = -1
REWARD = 3
DEFAULT_TRAINING = 5000

# User interface theme
CHARS = {"0": " ", "1": "X", "2": "O"}
logging.basicConfig(format="%(message)s")


# Command line interface
parser = argparse.ArgumentParser(
    description="Play tic-tac-toe against a reinforcement learning agent"
)

parser.add_argument("brainfile1", metavar="FILE", help="brain file to play against")
parser.add_argument(
    "-t",
    "--train",
    metavar="FILE",
    dest="brainfile2",
    help="second brainfile to play computer against computer",
)
parser.add_argument(
    "-T",
    "--train-both",
    action="store_true",
    help="update both brainfiles after training",
)
parser.add_argument(
    "-n",
    "--iterations",
    metavar="INT",
    type=int,
    default=DEFAULT_TRAINING,
    help=f"how many matches to play during training, default is {DEFAULT_TRAINING}",
)
parser.add_argument(
    "-q", "--quiet", action="store_true", help="don't print while training"
)
parser.add_argument(
    "-g", "--generate", action="store_true", help="write an empty brainfile to FILE"
)
parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="overwrite without asking when generating base brainfiles",
)


# Base brain generator
def brain_map():
    brain_map = {}
    for key in itertools.product("012", repeat=9):
        key = "".join(key)
        good_spots = [x == "0" for x in key]
        brain_map[key] = np.ones((9,), dtype=int) * good_spots
        brain_map[key] *= START_WEIGHT
    return brain_map


# Input Output
def print_state(state):
    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    print()
    print(" | ".join([CHARS[state[x]] for x in [6, 7, 8]]))
    print("---------")
    print(" | ".join([CHARS[state[x]] for x in [3, 4, 5]]))
    print("---------")
    print(" | ".join([CHARS[state[x]] for x in [0, 1, 2]]))
    print()


def parse_move_linear(instr):
    try:
        x = int(instr)
    except ValueError:
        raise
    if x not in range(1, 10):
        raise ValueError
    return x - 1


def load_brain(brainfile):
    try:
        with open(brainfile, "rb") as data:
            brainmap = pickle.load(data)
    except FileNotFoundError:
        logging.critical(f"File {brainfile} not found")
        exit(1)
    except pickle.UnpicklingError:
        logging.critical(f"Malformed brainfile in {brainfile}")
        exit(1)
    return brainmap


def update_brain(brain_map, moves, delta):
    for state, move in moves:
        # The following check should prevent death
        if delta > 0 or brain_map[state][move] > 1:
            brain_map[state][move] += delta


# Game handling
def game_result(state):
    # Tie: player 2 wins
    for i in range(3):
        row = state[i * 3 : i * 3 + 3]
        if row == "111":
            return 1
        if row == "222":
            return 2
    for j in range(3):
        col = state[j : 7 + j : 3]
        if col == "111":
            return 1
        if col == "222":
            return 2
    main_diag = state[0] + state[4] + state[8]
    if main_diag == "111":
        return 1
    if main_diag == "222":
        return 2
    second_diag = state[2] + state[4] + state[6]
    if second_diag == "111":
        return 1
    if second_diag == "222":
        return 2
    if "0" not in state:
        assert state.count("1") == state.count("2") - 1
        return 1
    return 0


def cogitate(state, brain_map):
    distribution = brain_map[state] / sum(brain_map[state])
    return np.random.choice(9, 1, p=distribution)[0]


# Playing and training
def self_play(brain_map1, brain_map2):
    state = "000000000"
    moves1 = []
    moves2 = []
    moves = [moves1, moves2]
    print_state(state)
    while True:
        # Computer 1 moves
        cmov = cogitate(state, brain_map1)
        assert state[cmov] == "0"
        moves1.append((state, cmov))
        state = state[:cmov] + "2" + state[cmov + 1 :]
        print_state(state)
        result = game_result(state)
        if result != 0:
            break
        # Computer 2 moves
        cmov = cogitate(state, brain_map2)
        assert state[cmov] == "0"
        moves2.append((state, cmov))
        state = state[:cmov] + "1" + state[cmov + 1 :]
        print_state(state)
        result = game_result(state)
        if result != 0:
            break
    print_state(state)
    if result == 1:
        logging.debug("Computer 2 wins!")
        return True, moves
    else:
        logging.debug("Computer 1 wins!")
        return False, moves


def self_train(brainmap1, brainmap2, n):
    for i in range(n):
        result, moves = self_play(brainmap1, brainmap2)
        if result:
            update_brain(brainmap2, moves[1], REWARD)
            update_brain(brainmap1, moves[0], PENALTY)
        else:
            update_brain(brainmap2, moves[1], PENALTY)
            update_brain(brainmap1, moves[0], REWARD)


def play_human(brain_map):
    state = "000000000"
    moves = []
    print_state(state)
    while True:
        # Human's move
        move = input(">>> ")
        try:
            hmov = parse_move_linear(move)
        except ValueError:
            print("Invalid input, please enter number 1-9")
            continue
        if state[hmov] == "0":
            state = state[:hmov] + "2" + state[hmov + 1 :]
        else:
            print("Cell already taken")
            continue
        result = game_result(state)
        if result != 0:
            break
        # Computer's move
        cmov = cogitate(state, brain_map)
        assert state[cmov] == "0"
        moves.append((state, cmov))
        state = state[:cmov] + "1" + state[cmov + 1 :]
        print_state(state)
        result = game_result(state)
        if result != 0:
            break
    print_state(state)
    if result == 1:
        print("Computer wins!")
        return True, moves
    else:
        print("You win!")
        return False, moves


if __name__ == "__main__":
    args = parser.parse_args()

    if args.generate:
        if not args.force and os.path.isfile(args.brainfile1):
            print(f"Brainfile {args.brainfile1} already exists")
            answer = input("Overwrite? [y/N] ")
            if answer != "y":
                print("Quitting")
                exit(0)
        bmap = brain_map()
        with open(args.brainfile1, "wb") as f:
            print("Saving base brain in {}".format(args.brainfile1))
            pickle.dump(bmap, f)
        exit(0)

    if args.quiet:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    brainmap1 = load_brain(args.brainfile1)
    computer_match = args.brainfile2 is not None
    training = args.train_both

    if computer_match:
        brainmap2 = load_brain(args.brainfile2)
        if training:
            n = args.iterations
            logging.info("Starting self training")
            start = time.monotonic()
            self_train(brainmap1, brainmap2, n)
            end = time.monotonic()
            logging.info(f"Trained {n} matches in {end - start:.3f} seconds")
            with open(args.brainfile1, "wb") as f:
                logging.info("Saving brain updates for computer 1")
                pickle.dump(brainmap1, f)
            with open(args.brainfile2, "wb") as f:
                logging.info("Saving brain updates for computer 2")
                pickle.dump(brainmap2, f)
        else:
            raise NotImplementedError
    else:
        logging.info("Playing human against computer")
        play_human(brainmap1)
