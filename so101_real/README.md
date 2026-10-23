# SO-101 VLA, real setup

Same pipeline as `../so101_vla/`, in a replica of your real setup (`../scene.xml`). A black SO-101 is mounted at the **center of the mat's front edge**, and the task is "Pick up the white block and place it on the green note."

The model sees the top and wrist RGB cameras, the task and the phase, and outputs the next joint keyframe, for example `j: -8.1 18.3 -35.6 95.0 12.5 | gripper: open`.

## Run order (from this folder, with `../.venv` active)

| Step | Command | Time |
|---|---|---|
| Watch the expert | `python real_env.py --no-randomize -n 3` | – |
| See the camera images | `python real_policy.py --policy expert --episodes 3` | – |
| Make a dataset | `python real_dataset.py --episodes 3000` | ~6 min, ~1.1 GB |
| Smoke test training | `python real_train.py --max-steps 30` | ~2 min |
| Train (1 epoch) | `python real_train.py` | ~2.5 h |
| Continue a stopped run | `python real_train.py --resume checkpoints/<run>` | – |
| Watch the model drive | `python real_policy.py --lora checkpoints/<run>/trainer/checkpoint-1000` | – |
| Success rate and why it fails | `python real_policy.py --lora checkpoints/<run>/lora --episodes 30 --headless` | ~15 s/episode |

Datasets go to `datasets/so101_real_v1_<timestamp>/` and training runs go to `checkpoints/qwen35_2b_so101_real_<timestamp>/`. Both are git-ignored.

## Building the real setup to match

All positions are measured from the **shoulder-pan axis on the table top**: +x points straight into the mat and +z points up.

- **Robot:** shoulder-pan axis at the center of the mat's front edge, facing into the mat.
- **Top camera:** 0.19 m ahead, 0.50 m above the table, looking straight down, with the top of the image pointing away from the robot. Vertical field of view 45°.
- **Wrist camera:** on the fixed jaw, as in `../so101_ik.py` (`WRIST_CAM_POS`, tilted 25° toward the jaws, 70° field of view).
- **Reach:** the arm grasps the 6.5 cm block between **20 and 32 cm** from its base (closer, it knocks the block over; farther, the wrist can't point straight down). Place the block in that band. The note's usual spot (from `scene.xml`) is about 32 cm away, so the expert aims up to 1.5 cm nearer the robot when the note is that far.

## Sim-to-real randomization

- **Per episode** (`real_env.py`): light strength and position; table, mat, note, block and arm color; top camera pose (about ±1 cm and 1.5°) and field of view.
- **Per image** (`real_dataset.py`): sensor noise, brightness and contrast, occasional slight blur.

`--no-randomize` turns all of this off and shows the nominal scene.

## Before running on the real arm

- Check that the sim's joint angles mean the same thing on the real arm: send the same angles to both and compare the poses. `so101_new_calib.xml` follows LeRobot's calibration. Check the gripper open (0.8 rad) and closed values too.
- The runner reports grasped / lifted / distance from the note / upright and the wrist rotation error. Use those to see why episodes fail, not just how many.
