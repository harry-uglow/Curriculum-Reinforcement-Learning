import rospy

from gym import Env

from std_msgs.msg import UInt16

from intera_interface import CHECK_VERSION, limb, RobotEnable


# TODO: Build as a Gym env with wrappers for policies
class RealEnv(Env):
    timestep = 0
    ep_len = 128

    def step(self, action):
        cmd = dict([(joint, action) for joint, action in zip(self._right_joint_names, action)])
        self._right_arm.set_joint_velocities(cmd)

        self.control_rate.sleep()

        self.timestep += 1
        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        return ob, 0, done, dict()

    def _get_obs(self):
        return [self._right_arm.joint_angle(joint_name)
                for joint_name in self._right_joint_names]

    def reset(self):
        stop_action = [0.] * len(self._right_joint_names)
        cmd = dict([(joint, action) for joint, action in zip(self._right_joint_names, stop_action)])
        self._right_arm.set_joint_velocities(cmd)
        self.timestep = 0

        input("Move the arm into open space and press Enter to continue.")

        # SETS TO A NEUTRAL POSITION. Remove from dish rack first.
        # TODO: Does this return once done or immediately?
        self.set_neutral()

        return self._get_obs()

    def render(self, mode='human'):
        pass

    def __init__(self):
        """
        'Wobbles' both arms by commanding joint velocities sinusoidally.
        """
        print("Initializing node... ")
        rospy.init_node("rsdk_dish_rack")
        rospy.on_shutdown(self.clean_shutdown)

        self._right_arm = limb.Limb("right")
        self._right_joint_names = self._right_arm.joint_names()  # TODO: Select relevant

        # control parameters
        self._rate = 20.0  # Hz
        self._missed_cmds = 3
        self.control_rate = rospy.Rate(self._rate)

        print("Getting robot state... ")
        self._rs = RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        self._init_joint_angles = [self._right_arm.joint_angle(joint_name)
                                   for joint_name in self._right_joint_names]
        rospy.set_param('named_poses/right/poses/neutral', self._init_joint_angles)

        self._right_arm.set_command_timeout((1.0 / self._rate) * self._missed_cmds)

    def set_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral pose...")
        self._right_arm.move_to_neutral()

    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        print("\nExiting example...")
        self._right_arm.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()

    def make_cmd(self, joint_names, action):
        return dict([(joint, action) for joint, action in zip(joint_names, action)])
