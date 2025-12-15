# Collecting Gesture Samples

Use `src/demo_cli.py` to preview hand landmarks. Extend it to save frames + labels:
- Press a key per gesture (e.g., `1` for NOTE_C4, `2` for NOTE_D4, etc.).
- Save the 21x2 landmarks and the chosen label into an `.npz` file in this folder.

Later, run a small training script to aggregate all `.npz` files, extract features, and train the classifier.