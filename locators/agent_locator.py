import collections
import logging
import os
import re
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, center_of_mass
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_dilation
from skimage.segmentation import watershed

DEBUG = False
SAVE_DIFFICULT = False
# if DEBUG:
#     logging.basicConfig(level=logging.DEBUG)
# else:
#     logging.basicConfig(level=logging.WARNING)

background = np.array(Image.open('data/background_64.png'))[:, :, :3]

red = np.array([151, 106, 1079])
blue = np.array([120, 128, 163])


class Agent(object):
    """Keeps track of an agent's position and velocity
    
    Attributes:
        pos (Optional[np.ndarray], optional): The position of the agent (in pixels). Defaults to None.
        vel (Optional[np.ndarray], optional): The velocity of the agent (in pixels/frame). Defaults to None.
    """

    def __init__(self, lag: float, lag_overlap: float, pos: Optional[np.ndarray] = None,
                 vel: Optional[np.ndarray] = None):
        """
        Args:
            pos (Optional[np.ndarray], optional): The initial position of the agent. Defaults to None.
            vel (Optional[np.ndarray], optional): The initial velocity of the agent. Defaults to None.
        """
        self.lag = lag
        self.lag_overlap = lag_overlap
        self.pos = pos
        self.pos_hist = collections.deque(maxlen=40)
        self.computed_pos = pos
        self.vel = vel

        # self.avg_vel = [0]

        self.last_pos_update: Optional[int] = None

    def update_pos(self, new_pos: np.ndarray, frame: int, overlap: bool) -> bool:
        """Updates the position of the agent given the newly computed position

        :param new_pos: The newly computed position
        :type new_pos: np.ndarray
        :param frame: The current frame index
        :type frame: int
        :param overlap: Whether the agent was overlapping with an other agent
        :type overlap: bool
        :return: Whether the agent accepts the new position
        :rtype: bool
        """
        if self.last_pos_update is not None:
            self.vel = (new_pos - self.computed_pos) / (frame - self.last_pos_update)
            # self.avg_vel += [np.linalg.norm(self.vel)]
            # print(np.mean(self.avg_vel), np.percentile(self.avg_vel, [50, 75, 99, 100]))
            # TODO: Make this threshold configurable
            if np.linalg.norm(self.vel) > 3.2:
                self.update_pos_based_on_vel(frame - self.last_pos_update)
                return False

        lag = self.lag_overlap if overlap else self.lag
        self.pos = (new_pos * (1 - lag) + self.pos * lag) if self.pos is not None else new_pos
        self.pos_hist.append(self.pos)
        self.computed_pos = new_pos
        self.last_pos_update = frame

        return True

    def update_pos_based_on_vel(self, frame_delta: int):
        """Updates the agent's position based on it's velocity. 

        Requires position and velocity to have previously computed values
        
        Args:
            frame_delta (int): The number of frames of velocity to compute
        
        Raises:
            ValueError: If either position or velocity is None
        """
        if self.vel is None or self.pos is None:
            raise ValueError('Position and/or velocity have not yet been initialized')
        vel = (self.pos_hist[-1] - self.pos_hist[0]) / (self.pos_hist.maxlen - 1)
        self.pos += vel * frame_delta
        self.pos_hist.append(self.pos)


def _grayness(c: np.ndarray) -> float:
    """Determines the level of grayness of a color

    The grayness is defines as follows:

    .. math:: \\frac{1}{(max(c)-min(c)/mean(c)}


    Parameters
    ----------
    c : np.ndarray
        The color to determine the level of grayness of

    Returns
    -------
    float
        The level of grayness
    """
    if np.mean(c) < 0.001:
        return 100000
    return 1 / ((np.max(c) - np.min(c)) / np.mean(c))


class AgentLocator(object):
    """Keeps track of agents based on either their previously recorded velocity or a state image
    
    Attributes:
        blue_agent (Agent): The blue agent position and velocity
        red_agent (Agent): The red agent position and velocity
        perspective (bool): whether to perspective transform the state image
    """

    def __init__(self, use_dilation: bool = False, use_erosion: bool = False, use_overlap_label_dilation: bool = True,
                 cooldown_time: int = 5,
                 minimum_agent_area: int = 4, minimum_agent_area_overlap: int = 3, blue_marker_thresh: int = 5,
                 red_marker_thresh: int = 5, lag: float = 0.1, lag_overlap: float = 0.4, save_difficult: bool = False,
                 save_difficult_prob: float = 0.001):
        """iwejifwej

        :param use_dilation: Whether to apply morphological dilation to the segmentation image
        :param use_erosion: Whether to apply morphological erosion the the marker images used by watershed
        :param use_overlap_label_dilation:
        :param cooldown_time: The number of frames to wait between agent localization from the provided state image,
            in between image localizations, the velocity is used to extrapolate the position
        :param minimum_agent_area: The minimum region size required to be recognized as an agent
        :param minimum_agent_area_overlap: The minimum region size required to be recognized as an agent, given that the
            agents are overlapping
        :param blue_marker_thresh: The least number of blue markers used to localize the blue agent using watershed
        :param red_marker_thresh: The least number of red markers used to localize the red agent using watershed
        :param lag: The lag of the agents with respect to newly computed positions
        :param lag_overlap: The lag of the agents with respect to newly computed positions, given that the agents are
            overlapping
        :param save_difficult: Whether to save frames that cause localization failures
        :param save_difficult_prob: The rate at which frames that cause localization failures are saved, only applies if
            ``save_difficult`` is true
        """
        self.blue_agent = Agent(lag, lag_overlap)
        self.red_agent = Agent(lag, lag_overlap)

        self.cooldown = 0
        self.cooldown_time = cooldown_time

        self.minimum_agent_area = minimum_agent_area
        self.minimum_agent_area_overlap = minimum_agent_area_overlap
        self.blue_marker_thresh = blue_marker_thresh
        self.red_marker_thresh = red_marker_thresh
        self.use_dilation = use_dilation
        self.use_erosion = use_erosion
        self.use_overlap_label_dilation = use_overlap_label_dilation

        self.save_difficult = save_difficult
        self.save_difficult_prob = save_difficult_prob

        self.errored = False
        self.frame_counter = 0
        self.size = 64
        self.perspective = False

    def update_agent_locations(self, state_img: np.ndarray):
        """Update the blue and red agents location based on a stage image

        Parameters
        ----------
        state_img : np.ndarray
            The state image from which to extract agent locations
        """
        self.errored = False
        # FIXME: Do not catch all exceptions...
        try:
            self.frame_counter += 1
            logging.debug(f'\nFrame {self.frame_counter}')
            if self.cooldown == 0 or self.blue_agent.vel is None or self.red_agent.vel is None:
                logging.debug('Computing agent locations based on image')
                foreground = np.mean(np.abs(state_img - background), axis=2)
                segmented = foreground > 0.01
                labeled = label(segmented if not self.use_dilation else binary_dilation(segmented, selem=np.array([
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]
                ])))
                regions = [r for r in regionprops(labeled) if r.area > self.minimum_agent_area]

                overlap_agents = state_img * np.repeat((labeled == regions[0].label)[:, :, np.newaxis], 3, axis=2)

                index = np.unravel_index(
                    np.argmax(overlap_agents[:, :, 2] - np.mean(overlap_agents[:, :, (0, 1)], axis=2)),
                    overlap_agents[:, :, 2].shape)
                blue = overlap_agents[index]
                index = np.unravel_index(np.argmax(overlap_agents[:, :, 0] - np.mean(overlap_agents[:, :, 1:], axis=2)),
                                         overlap_agents[:, :, 2].shape)
                red = overlap_agents[index]

                if len(regions) == 1:
                    # TODO: Make grayness threshold configurable
                    logging.debug(
                        f'Single region found, red grayness: {_grayness(red)}, blue grayness {_grayness(blue)}')
                    if _grayness(red) > 3.4:
                        logging.debug('Only blue agent detected')
                        self.blue_agent.update_pos(np.array(regions[0].centroid), self.frame_counter, overlap=False)
                        self.red_agent.update_pos_based_on_vel(1)
                        return
                    elif _grayness(blue) > 3.4:
                        logging.debug('Only red agent detected')
                        self.red_agent.update_pos(np.array(regions[0].centroid), self.frame_counter, overlap=False)
                        self.blue_agent.update_pos_based_on_vel(1)
                        return

                    logging.debug('Single, double agent region detected, segmenting...')

                    blue_img = np.mean(np.abs(overlap_agents.astype(np.float) - blue), axis=2)
                    red_img = np.mean(np.abs(overlap_agents.astype(np.float) - red), axis=2)

                    markers = np.zeros_like(blue_img)
                    for thr in np.linspace(0, 50, 100):
                        markers[binary_erosion(blue_img < thr) if self.use_erosion else blue_img < thr] = 1
                        if np.sum(markers == 1) > self.blue_marker_thresh:
                            break
                    for thr in np.linspace(0, 50, 100):
                        markers[binary_erosion(red_img < thr) if self.use_erosion else red_img < thr] = 2
                        if np.sum(markers == 2) > self.red_marker_thresh:
                            break

                    labeled = watershed(-distance_transform_edt(segmented), markers=markers, mask=segmented)

                    if np.sum(labeled == 1) == 0 or np.sum(labeled == 2) == 0:
                        self.errored = True
                        if DEBUG:
                            Image.fromarray(markers.astype(np.uint8)).save('error_markers.png')
                            raise ValueError(f'Found {len(regions)} regions, must be 2')
                        else:
                            logging.warning('Failed to segment double agent region, using velocity instead')
                            self._save_difficult(state_img, labeled, markers)
                            self._update_agent_locations_vel()
                            return

                    fail_sum = 0

                    # FIXME: This is unnecessary
                    if (np.min(np.mean(np.abs(
                            (overlap_agents * np.repeat((labeled == 2)[:, :, np.newaxis], 3, axis=2)).astype(
                                np.float) - red), axis=2)) <
                            np.min(np.mean(np.abs(
                                (overlap_agents * np.repeat((labeled == 2)[:, :, np.newaxis], 3, axis=2)).astype(
                                    np.float) - blue), axis=2))):
                        fail_sum += self.blue_agent.update_pos(np.array(center_of_mass(labeled == 1)),
                                                               self.frame_counter, overlap=True)
                        fail_sum += self.red_agent.update_pos(np.array(center_of_mass(labeled == 2)),
                                                              self.frame_counter, overlap=True)
                    else:
                        fail_sum += self.blue_agent.update_pos(np.array(center_of_mass(labeled == 2)),
                                                               self.frame_counter, overlap=True)
                        fail_sum += self.red_agent.update_pos(np.array(center_of_mass(labeled == 1)),
                                                              self.frame_counter, overlap=True)

                    if fail_sum > 0:
                        logging.debug('At least one agent detected error in localization')
                        self._save_difficult(state_img, labeled, markers)
                else:
                    logging.debug('Two separate regions found')
                    if np.argmax(np.mean(state_img[labeled == regions[1].label], axis=0)) == 0:
                        blue_region = regions[0]
                        red_region = regions[1]
                    else:
                        blue_region = regions[1]
                        red_region = regions[0]

                    self.blue_agent.update_pos(np.array(blue_region.centroid), self.frame_counter, overlap=False)
                    self.red_agent.update_pos(np.array(red_region.centroid), self.frame_counter, overlap=False)

                self.cooldown = self.cooldown_time
            else:
                logging.debug('Not extracting this frame')
                self._update_agent_locations_vel()
                self.cooldown -= 1
        except Exception as e:
            logging.debug(f'Exception {e} occurred')
            self.errored = True
            self._save_difficult(state_img, labeled, None)
            if DEBUG:
                Image.fromarray(labeled.astype(np.uint8)).save('error_labeled.png')
                Image.fromarray(state_img.astype(np.uint8)).save('error.png')
                raise ValueError()
            else:
                self._update_agent_locations_vel()

    def get_locations(self, state_img):
        """Determine the x and y pixel coordinates of the red and blue agents.

        Args:
            state_img = [ndarray] current state of environment as NumPy ndarray

        Returns [(x_red, y_red), (x_blue, y_blue)]:
            The x and y pixel coordinates of the red and blue agents
            as a tuple of NumPy ndarrays.
        """
        state = np.array(state_img).reshape(self.size, self.size, 3)

        self.update_agent_locations(state)
        return self.red_agent.pos, self.blue_agent.pos

    def _update_agent_locations_vel(self):
        """
        Update both agents' locations based on their current velocity
        """
        logging.debug('Updating agent locations based on velocity')
        self.blue_agent.update_pos_based_on_vel(1)
        self.red_agent.update_pos_based_on_vel(1)

    def _save_difficult(self, state_img: np.ndarray, labeled: np.ndarray, markers: np.ndarray):
        """
        Saves a state image to the difficult state directory, as well as the label image and the markers
        image.

        Parameters
        ----------
        state_img : np.ndarray (size, size, 3)
            The difficult state image to save
        labeled : np.ndarray (size, size)
            The computed label image associated with `state_img`
        markers : np.ndarray (size, size)
            The computed marker image associated with `state_img`
        """
        if self.save_difficult_prob and np.random.random() < self.save_difficult_prob:
            i = max([int(re.findall(r'(\d+)\.png', f.name)[0]) for f in os.scandir('difficult_states') if
                     re.match(r'\d+\.png', f.name)]) + 1
            if markers is not None:
                Image.fromarray(markers.astype(np.uint8)).save(f'difficult_states/{i}_markers.png')
            Image.fromarray(labeled.astype(np.uint8)).save(f'difficult_states/{i}_labeled.png')
            Image.fromarray(state_img.astype(np.uint8)).save(f'difficult_states/{i}.png')
