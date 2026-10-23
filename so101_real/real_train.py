"""Train on the real-setup dataset. Same training as ../so101_vla/train.py (all its options work here);
only the folders and the default run name differ: datasets/ and checkpoints/ inside so101_real/.

    python real_train.py --max-steps 30                 # smoke test
    python real_train.py                                # full run (1 epoch)
    python real_train.py --resume checkpoints/<run>     # continue an interrupted run
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(HERE), "so101_vla"))
import train as so101_train  # noqa: E402  (imports unsloth first, sets UNSLOTH_BLOCK_COMPILE=0)

so101_train.HERE = HERE
so101_train.RUNNER = "real_policy.py"
if "--name" not in sys.argv and "--resume" not in sys.argv:
    sys.argv += ["--name", "qwen35_2b_so101_real"]

if __name__ == "__main__":
    so101_train.main()
