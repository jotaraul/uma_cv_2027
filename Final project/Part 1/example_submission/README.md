# Example submission (Part 1)

**Members:** Name Surname, Name Surname

This is an example that follows the format of a submission. Its detector is
deliberately naive, so expect a very low score: use it to see how a
submission is organized, not as a starting point for your method.

## Method

A color threshold. Pixels that are saturated and have a warm hue (red to
yellow-green) are marked as fruit. The marked pixels are cleaned with a
morphological opening and split into connected regions, and every region of
at least 40 pixels becomes one detection. Its species comes from the median
hue of the region (red: apple, orange: orange, yellow: lemon, yellow-green:
pear), and its score is the mean saturation.

## Training

None: the thresholds were set by hand.

## Validation scores

Scores on the whole train set, with `evaluate.py`: see the table that
`check_submission.py` prints.

## External resources

None beyond `numpy`, `pillow`, `scipy` and `pycocotools`.
