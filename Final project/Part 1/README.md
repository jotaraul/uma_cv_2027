# Final project, Part 1: starter kit

Tools for Part 1 of the final project, *Detecting and counting fruits on
trees*. The full description of the task is on the Part 1 page of the Virtual
Campus. Download the train set, `fruits_part1.zip`, from the Virtual
Campus too.

| File | Use |
|---|---|
| `evaluate.py` | Scores a predictions file against the annotations, with the official metrics |
| `check_submission.py` | Checks your submission the way we run it, before you submit it |
| `visualize.py` | Draws annotations or predictions over the images |
| `example_submission/` | A minimal submission that follows the rules. Its detector is deliberately naive: it shows the format, not a method |

## Installation

Any recent Python 3 works, but we run submissions with Python 3.12, so using
it too avoids surprises:

```bash
python3.12 -m venv .venv                # Linux and macOS
source .venv/bin/activate
pip install -r requirements.txt
```

```bat
py -3.12 -m venv .venv                  :: Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Score predictions (the whole train set, or some images, e.g. a validation split):

```bash
python evaluate.py --gt fruits_part1/train/annotations.json --pred predictions.json
python evaluate.py --gt fruits_part1/train/annotations.json --pred predictions.json --images-file val_ids.txt
```

Look at the annotations, or at your predictions:

```bash
python visualize.py --images fruits_part1/train/images --coco fruits_part1/train/annotations.json --out vis/
python visualize.py --images fruits_part1/train/images --coco fruits_part1/train/annotations.json --out vis/ --limit 10 --no-labels
python visualize.py --images fruits_part1/train/images --pred predictions.json --images-json fruits_part1/train/images.json --min-score 0.5 --out vis_pred/
```

Run the example submission on the train set and score it:

```bash
cd example_submission
pip install -r requirements.txt
python main.py detect --input ../fruits_part1/train --output ../predictions.json --device cpu
cd ..
python evaluate.py --gt fruits_part1/train/annotations.json --pred predictions.json
```

Check your submission before submitting it. It installs it in a new virtual
environment and runs it on 10 images chosen at random from the train set given
with `--data`, then scores them. It does not disable the internet, so make sure
yourself that your program does not download anything: on our machine, it runs
without internet access.

```bash
python check_submission.py team_<name>/ --data fruits_part1/train
python check_submission.py team_<name>/ --data fruits_part1/train --device cpu    # without a GPU
```
