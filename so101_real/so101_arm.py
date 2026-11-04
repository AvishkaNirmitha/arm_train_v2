"""The real SO-101 follower arm, driven through lerobot with only the motors attached so far.

lerobot's `so101_follower` expects all 6 motors to answer when it connects. This wraps it with a motor list,
ATTACHED, so the arm runs without wrist_roll (taped, not driven) until it is mounted. Its calibration file lives
under robots/so101_partial (not so101_follower), so the stock robot never picks up a partial calibration.

Run in .venv-lerobot (Python 3.12), from the project root:

    python so101_real/so101_arm.py setup                            # give ids to the ATTACHED motors, one at a time
    python so101_real/so101_arm.py setup --motors wrist_roll         # later: only the new motor
    python so101_real/so101_arm.py calibrate                        # all ATTACHED motors
    python so101_real/so101_arm.py calibrate --motors gripper       # only these; the others keep their calibration
    python so101_real/so101_arm.py read                             # torque off, print joint angles (deg)

Adding a motor later: `setup --motors wrist_roll` (alone on the board), mount and chain it, add it to ATTACHED,
then `calibrate --motors wrist_roll`: recalibrating the others would move their zeros (real_joint_map.json).
"""
import time
import argparse

from lerobot.motors import SerialMotorsBus
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

ALL_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")  # ids 1..6
ATTACHED = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "gripper")  # add "wrist_roll" when mounted

PORT = "/dev/ttyACM0"
ROBOT_ID = "so101_arm"


class SO101Arm(SOFollower):
    name = "so101_partial"

    def __init__(self, config, motors=ATTACHED):
        super().__init__(config)
        # The bus builds its id tables on creation: make a new one with only these motors
        self.calibration = {m: c for m, c in self.calibration.items() if m in motors}
        self.bus = SerialMotorsBus(port=config.port, motors={m: self.bus.motors[m] for m in motors},
                                   calibration=self.calibration)

    def configure(self):
        # Goal := present position before configure() switches torque on, so nothing jumps (raw steps: works
        # for a motor that isn't calibrated yet)
        self.bus.disable_torque()
        self.bus.sync_write("Goal_Position", self.bus.sync_read("Present_Position", normalize=False),
                            normalize=False)
        super().configure()

    def calibrate(self, motors=None):
        """motors: only these are recalibrated, the others keep their calibration from the file."""
        if motors is None:
            # A file from before a motor was added doesn't cover it: recalibrate instead of offering the file
            if set(self.bus.motors) <= set(self.calibration) and self._keep_calibration_file(self.bus):
                return
            motors = list(self.bus.motors)
        elif missing := [m for m in self.bus.motors if m not in motors and m not in self.calibration]:
            raise SystemExit(f"{missing} have no calibration yet: add them to --motors")
        print("Video walkthrough: https://huggingface.co/docs/lerobot/main/en/so101#calibration-video")
        full_turn = [m for m in ("wrist_roll",) if m in motors]
        new = self.bus.record_calibration("position", full_turn_motors=full_turn, motors=motors)
        self.calibration = {m: new[m] if m in new else self.calibration[m] for m in self.bus.motors}
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)


def make_arm(port=PORT, robot_id=ROBOT_ID, motors=ATTACHED):
    return SO101Arm(SOFollowerRobotConfig(port=port, id=robot_id), motors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["setup", "calibrate", "read"])
    ap.add_argument("--motors", nargs="+", choices=ALL_MOTORS,
                    help="setup: these motors (default ATTACHED), in this order; calibrate: only these")
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--id", default=ROBOT_ID)
    args = ap.parse_args()

    if args.command == "setup":
        # None: like lerobot-setup-motors, from the end of the arm down to the base
        make_arm(args.port, args.id, args.motors or ATTACHED).bus.setup_motors(args.motors)
        return

    arm = make_arm(args.port, args.id)
    arm.connect(calibrate=False)
    try:
        if args.command == "calibrate":
            arm.calibrate(args.motors)
        else:
            if not arm.is_calibrated:
                raise SystemExit("Not calibrated: run `python so101_real/so101_arm.py calibrate` first.")
            arm.bus.disable_torque()   # move the arm by hand
            while True:
                obs = arm.get_observation()
                print("  ".join(f"{m}: {obs[f'{m}.pos']:7.1f}" for m in arm.bus.motors), end="\r")
                time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
