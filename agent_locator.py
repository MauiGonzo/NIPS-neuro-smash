import logging
from typing import Optional
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
from skimage.segmentation import watershed

DEBUG = False
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)

background = np.array(Image.open('background.png'))[:, :, :3]

red = np.array([217, 66, 59])
blue = np.array([0, 114, 201])

class Agent(object):
    """Keeps track of an agent's position and velocity
    
    Attributes:
        pos (Optional[np.ndarray], optional): The position of the agent (in pixels). Defaults to None.
        vel (Optional[np.ndarray], optional): The velocity of the agent (in pixels/frame). Defaults to None.
    """    
    def __init__(self, pos: Optional[np.ndarray] = None, vel: Optional[np.ndarray] = None):
        """
        Args:
            pos (Optional[np.ndarray], optional): The initial position of the agent. Defaults to None.
            vel (Optional[np.ndarray], optional): The initial velocity of the agent. Defaults to None.
        """                             
        self.pos = pos
        self.computed_pos = pos
        self.vel = vel

        self.last_pos_update: Optional[int] = None
    
    def update_pos(self, new_pos: np.ndarray, frame: int):
        """Updates the position of the agent (assumed to be computed accurately from a state_image)
        
        Args:
            new_pos (np.ndarray): The new position to set
            frame (int): The frame at which the position was computed
        """        
        if self.last_pos_update is not None:
            self.vel = (new_pos - self.computed_pos)/(frame - self.last_pos_update)

        self.pos = new_pos
        self.computed_pos = new_pos
        self.last_pos_update = frame
    
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
        self.pos += self.vel*frame_delta

class AgentLocator(object):
    """Keeps track of agents based on either their previously recorded velocity or a state image
    
    Attributes:
        blue_agent (Agent): The blue agent position and velocity
        red_agent (Agent): The red agent position and velocity
    """     

    def __init__(self, cooldown_time: int = 5, minimum_agent_area: int = 400, blue_marker_thresh: int = 150, red_marker_thresh: int = 150):
        """
        Args:
            cooldown_time (int) -- The time (in frames) between location updates based on the state image, rather than velocity (default: {5})
            minimum_agent_area (int) -- The minimum area of a region to be considered an agent(default: {400})
            blue_marker_thresh (int) -- The number of blue markers required for watershed segmentation (default: {150})
            red_marker_thresh (int) -- The number of red markers required for watershed segmentation (default: {150})
        """
        self.blue_agent = Agent()
        self.red_agent = Agent()

        self.cooldown = 0
        self.cooldown_time = cooldown_time

        self.minimum_agent_area = minimum_agent_area
        self.blue_marker_thresh = blue_marker_thresh
        self.red_marker_thresh = red_marker_thresh

        self.frame_counter = 0
    
    def update_agent_locations(self, state_img: np.ndarray):
        """Updates the blue and red agent's locations and velocity
        
        Args:
            state_img (np.ndarray): The current frame's state image
        """
        self.frame_counter += 1
        logging.debug(f'\nFrame {self.frame_counter}')
        if self.cooldown == 0 or self.blue_agent.vel is None or self.red_agent.vel is None:
            logging.debug('Computing agent locations based on image')
            foreground = np.mean(np.abs(state_img-background), axis=2)
            labeled = label(foreground > 0.01)
            regions = [r for r in regionprops(labeled) if r.area > self.minimum_agent_area]
            if len(regions) == 1:
                logging.debug('Single agent region detected, segmenting...')
                overlap_agents = state_img * np.repeat((labeled == regions[0].label)[:, :, np.newaxis], 3, axis=2)
                blue_img = np.mean(np.abs(overlap_agents.astype(np.float) - blue), axis=2)
                red_img = np.mean(np.abs(overlap_agents.astype(np.float) - red), axis=2)
                
                markers = np.zeros_like(blue_img)
                for thr in range(3, 50):
                    markers[binary_erosion(blue_img < thr)] = 1
                    if np.sum(markers == 1) > self.blue_marker_thresh:
                        break
                for thr in range(3, 50):
                    markers[binary_erosion(red_img < thr)] = 2
                    if np.sum(markers == 2) > self.red_marker_thresh:
                        break

                labeled = watershed(np.mean(overlap_agents, axis=2), markers, mask=(labeled == regions[0].label))
                regions = regionprops(labeled)
                regions = [r for r in regions if r.area > self.minimum_agent_area]
                
                if len(regions) != 2:
                    if DEBUG:
                        Image.fromarray(markers.astype(np.uint8)).save('error_markers.png')
                        Image.fromarray(labeled.astype(np.uint8)).save('error_labeled.png')
                        Image.fromarray(state_img.astype(np.uint8)).save('error.png')
                        raise ValueError(f'Found {len(regions)} regions, must be 2')
                    else:
                        logging.warning('Failed to segment double agent region, using velocity instead')
                        self._update_agent_locations_vel()
                        return
            
            if np.argmax(np.mean(state_img[labeled == regions[1].label], axis=0)) == 0:
                blue_region = regions[0]
                red_region = regions[1]
            else:
                blue_region = regions[1]
                red_region = regions[0]

            self.blue_agent.update_pos(np.array(blue_region.centroid), self.frame_counter)
            self.red_agent.update_pos(np.array(red_region.centroid), self.frame_counter)

            self.cooldown = self.cooldown_time
        else:
            self._update_agent_locations_vel()
            self.cooldown -= 1
    
    def _update_agent_locations_vel(self):
        logging.debug('Updating agent locations based on velocity')
        self.blue_agent.update_pos_based_on_vel(1)
        self.red_agent.update_pos_based_on_vel(1)
