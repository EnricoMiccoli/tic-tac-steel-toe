import argparse
import itertools
import logging
import numpy as np
import os
import pickle
import time

EMPTY_BOARD = "000000000"

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
    description=("Play tic-tac-toe against a reinforcement learning agent."
        " If no brainfile is specified for a given player,"
        " then that player is assumed to be a human."
        )
)
parser.add_argument(
    "-p1", "--player1", metavar="P1", help="brainfile for the first player"
)
parser.add_argument(
    "-p2", "--player2", metavar="P2", help="brainfile for the second player"
)
parser.add_argument(
    "-t1", "--train1", metavar="P1", help="brainfile to train as a first player"
)
parser.add_argument(
    "-t2", "--train2", metavar="P2", help="brainfile to train as a second player"
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
    "-g", "--generate", metavar="FILE", help="write an empty brainfile to FILE"
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


def print_tutorial():
    print()
    print("7 | 8 | 9 ")
    print("---------")
    print("4 | 5 | 6 ")
    print("---------")
    print("1 | 2 | 3 ")
    print()


def parse_move_linear(instr):
    try:
        x = int(instr)
    except ValueError:
        raise
    if x not in range(1, 10):
        raise ValueError
    return x - 1


def read_human_move(state):
    while True:
        move = input(">>> ")
        try:
            hmov = parse_move_linear(move)
        except ValueError:
            print("Invalid input, please enter number 1-9")
            continue
        if state[hmov] == "0":
            return hmov
        else:
            print("Cell already taken")
            continue


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


def make_human_move(player_id, state):
    hmov = read_human_move(state)
    assert state[hmov] == "0"
    updated = state[:hmov] + player_id + state[hmov + 1 :]
    return updated


def make_computer_move(player_id, brainmap, state):
    cmov = cogitate(state, brainmap)
    assert state[cmov] == "0"
    updated = state[:cmov] + player_id + state[cmov + 1 :]
    return cmov, updated


# Playing and training
def self_play(brain_map1, brain_map2):
    state = "000000000"
    moves = [[], []]
    print_state(state)
    result = 0
    while result == 0:
        iterators = zip(["2", "1"], [brain_map1, brain_map2], moves)
        for player_id, brainmap, mv in iterators:
            move, updated = make_computer_move(player_id, brainmap, state)
            mv.append((state, move))
            state = updated
            print_state(state)
            result = game_result(state)
            if result != 0:
                break
    if result == 1:
        logging.debug("Computer 2 wins!")
        return True, moves
    else:
        assert result == 2
        logging.debug("Computer 1 wins!")
        return False, moves


def self_train(brainmap1, brainmap2, n, training1, training2):
    for i in range(n):
        result, moves = self_play(brainmap1, brainmap2)
        if result:
            update_brain(brainmap2, moves[1], REWARD) if training2 else None
            update_brain(brainmap1, moves[0], PENALTY) if training1 else None
        else:
            update_brain(brainmap2, moves[1], PENALTY) if training2 else None
            update_brain(brainmap1, moves[0], REWARD) if training1 else None


def play_human_computer(brainmap, humanfirst, training):
    if humanfirst:
        print("Playing human against computer")
    else:
        print("Playing computer against human")

    print_tutorial()
    state = EMPTY_BOARD
    mv = []

    # TODO use currying instead
    def _make_computer_move(player_id, state):
        return make_computer_move(player_id, brainmap, state)

    movers = [make_human_move, _make_computer_move]
    ts = [0, training]
    humans = [1, 0]
    if not humanfirst:
        movers.reverse()
        ts.reverse()
        humans.reverse()

    result = 0
    while result == 0:
        for player_id, mover, t, human in zip(["2", "1"], movers, ts, humans):
            update = mover(player_id, state)
            if human:
                state = update
            else:
                mv.append((state, update[0]))
                state = update[1]
            print_state(state)
            result = game_result(state)
            if result != 0:
                break

    if (result == 1 and humanfirst) or (result == 2 and not humanfirst):
        print("Computer wins!")
        update_brain(brainmap, mv, REWARD) if training else None
    else:
        print("You win!")
        update_brain(brainmap, mv, PENALTY) if training else None


def play_humans():
    state = "000000000"
    print("Playing human against human")
    print_tutorial()
    result = 0
    while result == 0:
        for player_id in ["2", "1"]:
            state = make_human_move(player_id, state)
            print_state(state)
            result = game_result(state)
            if result != 0:
                break
    if result == 1:
        print("X wins!")
    else:
        assert result == 2
        print("O wins!")


def play(p1, p2, t1, t2, n):  # path path bool bool int

    # Load files
    if p1 is not None:
        brainmap1 = load_brain(p1)
    if p2 is not None:
        brainmap2 = load_brain(p2)

    # Play
    if p1 is None and p2 is None:
        play_humans()
        return
    elif p1 is not None and p2 is not None:
        logging.info(f"Playing computer {p1} against computer {p2}")
        start = time.monotonic()
        self_train(brainmap1, brainmap2, n, t1, t2)
        end = time.monotonic()
        if t1 or t2:
            logging.info(f"Trained {n} matches in {end - start:.3f} seconds")
    elif p1 is not None:
        play_human_computer(brainmap1, 0, t1)
    elif p2 is not None:
        play_human_computer(brainmap2, 1, t2)
    else:
        assert False

    # Update
    if t1:
        with open(p1, "wb") as f:
            logging.info("Saving brain updates for computer 1")
            pickle.dump(brainmap1, f)
    if t2:
        with open(p2, "wb") as f:
            logging.info("Saving brain updates for computer 2")
            pickle.dump(brainmap2, f)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.generate is not None:
        if not args.force and os.path.isfile(args.generate):
            print(f"Brainfile {args.generate} already exists")
            answer = input("Overwrite? [y/N] ")
            if answer != "y":
                print("Quitting")
                exit(0)
        bmap = brain_map()
        with open(args.generate, "wb") as f:
            print(f"Saving base brain in {args.generate}")
            pickle.dump(bmap, f)
        exit(0)

    if args.quiet:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    p1 = args.player1 or args.train1
    p2 = args.player2 or args.train2
    t1 = args.train1 is not None
    t2 = args.train2 is not None

    play(p1, p2, t1, t2, args.iterations)
