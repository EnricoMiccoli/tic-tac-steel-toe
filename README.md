# tic-tac-steel-toe
Yet another reinforcement learning approach to solving tic-tac-toe. Like steel-toe boots: they're boots, but reinforced.

This is the most straightforward, unsophisticated, software implementation of a [MENACE](https://en.wikipedia.org/wiki/Donald_Michie#Career_and_research)-like system I could come up with.


## Usage
```
ttst.py [-h] [-t FILE] [-T] [-n INT] [-q] [-g] FILE

positional arguments:
  FILE                  brain file to play against

optional arguments:
  -h, --help            show this help message and exit
  -t FILE, --train FILE
                        second brainfile to play computer against computer
  -T, --train-both      update both brainfiles after training
  -n INT, --iterations INT
                        how many matches to play during training, default is
                        5000
  -q, --quiet           don't print while training
  -g, --generate        write an empty brainfile to FILE
  -f, --force           overwrite without asking when generating base
                        brainfiles
```

To run for the first time:
```
$ python ssts.py -g brain1.pickle
$ python ssts.py -g brain2.pickle
$ python ssts.py brain1.pickle -T -t brain2.pickle -n 10000
$ python ssts.py brain2.pickle
```
If you are winning most of the matches, try training with a higher `n`.

When you run `$ python ssts.py brain1.pickle -T -t brain2.pickle` you are training two models at the same time, brain1 will be a good first player, brain2 a good second player. A tie is considered a win for the second player.

