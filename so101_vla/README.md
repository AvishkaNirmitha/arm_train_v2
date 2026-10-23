# SO-101 VLA

Same pipeline as the Panda model (`../expert.py`, `../generate_dataset.py`, `../vlm_training/`), for the 5-DOF SO-101 arm. The scene and IK come from `../so101_ik.py`.

The model sees the top and wrist RGB cameras, the task and the phase, and outputs the next joint keyframe:

```
j: 11.7 -23.6 34.0 79.6 -16.9 | gripper: open
```

## Run order (from this folder, with `../.venv` active)

| Step | Command | Time |
|---|---|---|
| Watch the expert | `python so101_env.py` | – |
| Make a dataset | `python generate_dataset.py --episodes 3000` | ~7 min, ~1.1 GB |
| Smoke test training | `python train.py --max-steps 30` | ~2 min |
| Train (1 epoch) | `python train.py` | ~2.5 h |
| Continue a stopped run | `python train.py --resume checkpoints/<run>` | – |
| Watch the model drive | `python run_policy.py --lora checkpoints/<run>/trainer/checkpoint-1000` | – |
| Success rate | `python run_policy.py --lora checkpoints/<run>/lora --episodes 20 --headless` | ~20 s/episode |

Datasets go to `datasets/so101_v1_<dd-mm-yyyy-hh-mm-ss>/` and training runs go to `checkpoints/<name>_<timestamp>/`. Both are git-ignored.

## Same look as the Panda scene

`apply_panda_look()` in `so101_env.py` restyles the `so101_ik.py` scene when it loads: a flat dark-blue floor (no checkerboard, no reflection), the same headlight, overhead lights and haze, no shadows, and a green cube on a red pad with the same task text. Only the robot and the cube size (25 mm) differ.

## Differences from the Panda setup

- **5 joints + gripper.** The servos are stiff, so the arm tracks the commanded joints within about 0.1 mm and needs no gravity compensation.
- **Image-only prompt from the start.** The Panda model copied the "current joints" from its prompt and ignored the images.
- **Parked start pose.** At the IK home pose the arm sits in the middle of the top camera's view and hides the cube, so each episode starts with the base turned 100° to the side. Planning still seeds the IK from the home pose, so the labels don't depend on the parked pose.
- **Closer top camera** (0.5 m, 45° field of view), about 1.2 mm per pixel. The cube is only 25 mm wide.
- **Less tolerance.** With a 3 mm grasp error the expert succeeds about 90% of the time, and with 12 mm about 60%. So this model needs to be more precise than the Panda one (about 12 mm off).
