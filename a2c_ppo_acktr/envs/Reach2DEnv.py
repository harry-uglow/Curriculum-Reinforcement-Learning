# Python imports
import numpy as np
import pygame
from gym import spaces, Env


class Reach2DEnv(Env):
    def render(self, mode='human'):
        # Check for mouse events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        self.draw_current_state(self.joint_angles, self.target_pose)

    observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]),
                                   np.array([1, 1, 1, 1, 1]),
                                   dtype=np.float32)
    action_space = spaces.Box(np.array([0, 0, 0]),
                              np.array([1, 1, 1]), dtype=np.float32)
    joint_angles = np.array([0.1, 1.0, 0.5])
    target_pose = np.array([0.2, 0.2])
    link_lengths = [0.2, 0.15, 0.1]
    timestep = 0
    screen_size = 900

    def __init__(self, seed, rank, headless, ep_len=32):
        self.target_norm = self.normalise_target()
        self.np_random = np.random.RandomState()
        self.rank = rank
        self.np_random.seed(seed + rank)
        self.ep_len = ep_len
        if not headless:
            self.screen = pygame.display.set_mode((self.screen_size,
                                                   self.screen_size))
            pygame.display.set_caption("PPO Output Visualisation")

    def normalise_target(self, lower=-0.5, upper=0.5):
        return (self.target_pose - lower) / (upper - lower)

    def normalise_joints(self):
        js = self.joint_angles / np.pi
        rem = lambda x: x - int(x)
        return np.array(
            [rem((j + (abs(j) // 2 + 1.5) * 2) / 2.) for j in js])

    def unnormalise(self, dts):
        max_dt = np.pi / 18
        return np.array([(dt * 2 * max_dt) - max_dt for dt in dts])

    # Function to compute the transformation matrix between two frames, based on the length (in y-direction) along the first frame, and the rotation of the second frame with respect to the first frame
    def compute_transformation(self, length, rotation):
        cos_rotation = np.cos(rotation)
        sin_rotation = np.sin(rotation)
        transformation = np.array([[cos_rotation, -sin_rotation, 0],
                                   [sin_rotation, cos_rotation, length],
                                   [0, 0, 1]])
        return transformation

    # Function to compute the pose of end of the robot, based on the angles of all the joints
    def compute_end_pose(self):
        trans_10 = self.compute_transformation(0, self.joint_angles[0])
        trans_21 = self.compute_transformation(self.link_lengths[0], self.joint_angles[1])
        trans_32 = self.compute_transformation(self.link_lengths[1], self.joint_angles[2])
        trans_30 = np.dot(trans_10, np.dot(trans_21, trans_32))
        end_pose = np.dot(trans_30, np.array([0, self.link_lengths[2], 1]))
        return end_pose[0:2]

    def reset(self):
        self.target_pose = self.np_random.uniform(0, 0.3, 2)
        # self.target_pose = np.array([0.2, 0.2])
        self.target_norm = self.normalise_target()

        self.joint_angles = np.array([0.1, 1.0, 0.5])
        self.timestep = 0

        return self._get_obs()

    def step(self, a):
        joint_velocities = self.unnormalise(a)
        vec = self.compute_end_pose() - self.target_pose
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(joint_velocities).sum()
        reward = reward_dist + reward_ctrl

        self.joint_angles += joint_velocities
        self.timestep += 1

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)
        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl)

    def _get_obs(self):
        norm_joints = self.normalise_joints()
        return np.append(norm_joints, self.target_norm)

    # Function to draw the state of the world onto the screen
    def draw_current_state(self, current_joint_angles, current_target_pose):
        # First, compute the transformation matrices for the frames for the different joints
        trans_10 = self.compute_transformation(0, current_joint_angles[0])
        trans_21 = self.compute_transformation(self.link_lengths[0], current_joint_angles[1])
        trans_32 = self.compute_transformation(self.link_lengths[1], current_joint_angles[2])
        # Then, compute the coordinates of the joints in world space
        joint_1 = np.dot(trans_10, np.array([0, 0, 1]))
        joint_2 = np.dot(trans_10, (np.dot(trans_21, np.array([0, 0, 1]))))
        joint_3 = np.dot(trans_10, (np.dot(trans_21, np.dot(trans_32, np.array([0, 0, 1])))))
        robot_end = np.dot(trans_10, (np.dot(trans_21, np.dot(trans_32, np.array([0, self.link_lengths[2], 1])))))
        # Then, compute the coordinates of the joints in screen space
        joint_1_screen = [int((0.5 + joint_1[0]) * self.screen_size), int((0.5 - joint_1[1]) * self.screen_size)]
        joint_2_screen = [int((0.5 + joint_2[0]) * self.screen_size), int((0.5 - joint_2[1]) * self.screen_size)]
        joint_3_screen = [int((0.5 + joint_3[0]) * self.screen_size), int((0.5 - joint_3[1]) * self.screen_size)]
        robot_end_screen = [int((0.5 + robot_end[0]) * self.screen_size), int((0.5 - robot_end[1]) * self.screen_size)]
        target_screen = [int((0.5 + current_target_pose[0]) * self.screen_size), int((0.5 - current_target_pose[1]) * self.screen_size)]
        # Finally, draw the joints and the links
        self.screen.fill((0, 0, 0))
        pygame.draw.circle(self.screen, (255, 0, 0), target_screen, 15)
        pygame.draw.line(self.screen, (255, 255, 255), joint_1_screen, joint_2_screen, 5)
        pygame.draw.line(self.screen, (255, 255, 255), joint_2_screen, joint_3_screen, 5)
        pygame.draw.line(self.screen, (255, 255, 255), joint_3_screen, robot_end_screen, 5)
        pygame.draw.circle(self.screen, (0, 0, 255), joint_1_screen, 10)
        pygame.draw.circle(self.screen, (0, 0, 255), joint_2_screen, 10)
        pygame.draw.circle(self.screen, (0, 0, 255), joint_3_screen, 10)
        pygame.draw.circle(self.screen, (0, 255, 0), robot_end_screen, 10)
        pygame.display.flip()
