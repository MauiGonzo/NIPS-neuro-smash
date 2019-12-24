import math
import torch


class Aggregator(object):
    """Class that is used to compute the input for the agents.

    Attributes:
        size                     = [int] width and height pixels of environment
        time_delta               = [float] time between two frames in seconds
        diagonal                 = [float] length of the diagonal of the stage
        center                   = [(float, float)] location of the center
        num_obs                  = [int] number of elements in aggregate vector
        device                   = [torch.device] device to put the vector on
        old_red_new_red_dir_idx  = [int] vector index of direction from the
                                         old to the new red agent locations
        new_red_new_blue_dir_idx = [int] vector index of direction from the
                                         new red to the new blue agent locations
    """

    def __init__(self, size, time_delta=0.2, device=torch.device('cpu')):
        """Initializes the object with the meta parameters.

        Args:
            size       = [int] width and height pixels of environment
            time_delta = [float] time between two frames in seconds
            device     = [torch.device] device to put the vector on
        """
        self.size = size
        self.time_delta = time_delta

        self.diagonal = math.hypot(size, size)
        self.center = size/2, size/2

        self.num_obs = 18  # number of elements in aggregate vector
        self.device = device

        # set indices used by ChaseAgent
        self.old_red_new_red_dir_idx = 4
        self.new_red_new_blue_dir_idx = 13

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Compute the Euclidean distance, in pixels,
           between locations 1 and 2.

        Args:
            x1 = [float] the x pixel coordinate of the first location
            y1 = [float] the y pixel coordinate of the first location
            x2 = [float] the x pixel coordinate of the second location
            y2 = [float] the y pixel coordinate of the second location

        Returns [float]:
            The distance between locations 1 and 2.
        """
        return math.hypot(x1-x2, y1-y2)

    @staticmethod
    def direction(x1, y1, x2, y2):
        """Compute the direction from location 1 to 2 as the counterclockwise
           angle between the vector from (x, y)=(0, 0) to (x, y)=(0, 1)
           and the direction vector in the range (-pi, pi].

        Args:
            x1 = [float] the x pixel coordinate of the first location
            y1 = [float] the y pixel coordinate of the first location
            x2 = [float] the x pixel coordinate of the second location
            y2 = [float] the y pixel coordinate of the second location

        Returns [float]:
            The direction from location 1 to 2 in (-pi, pi].
        """
        rad = math.atan2(y1-y2, x1-x2)
        if rad <= -math.pi*1/2:  # deg <= -90 degrees
            return rad + math.pi*3/2  # deg + 270 degrees
        else:
            return rad - math.pi*1/2  # deg - 90 degrees

    def speed(self, x1, y1, x2, y2):
        """Compute the speed of the agent between locations 1 and 2
           in pixels per second.

        Args:
            x1 = [float] the x pixel coordinate of the first location
            y1 = [float] the y pixel coordinate of the first location
            x2 = [float] the x pixel coordinate of the second location
            y2 = [float] the y pixel coordinate of the second location

        Returns [float]:
            The speed of the agent.
        """
        return self.distance(x1, y1, x2, y2) / self.time_delta

    def dist_from_center(self, x, y):
        """Compute the Euclidean distance, in pixels, between the location
           and the center of the stage.

        Args:
            x = [float] the x pixel coordinate of the location
            x = [float] the y pixel coordinate of the location

        Returns [float]:
            The distance between the location and the center of the stage.
        """
        return self.distance(x, y, self.center[0], self.center[1])

    def dist_from_edge(self, x, y):
        """Compute the Euclidean distance, in pixels, between the location
           and the edge of the stage.

        Args:
            x = [float] the x pixel coordinate of the location
            x = [float] the y pixel coordinate of the location

        Returns [float]:
            The distance between the location and the edge of the stage.
        """
        return min(x, (self.size + 1) - x, y, (self.size + 1) - y)

    def further_from_center(self, x_red, y_red, x_blue, y_blue):
        """Determine whether the red agent is further away from the center
           of the stage than the blue agent.

        Args:
            x_red  = [float] the x pixel coordinate of the red agent
            y_red  = [float] the y pixel coordinate of the red agent
            x_blue = [float] the x pixel coordinate of the blue agent
            y_blue = [float] the y pixel coordinate of the blue agent

        Returns [bool]:
            Whether the red agent is further away from the center than
            the blue agent.
        """
        dist_from_center_red = self.dist_from_center(x_red, y_red)
        dist_from_center_blue = self.dist_from_center(x_blue, y_blue)

        return dist_from_center_red > dist_from_center_blue

    def close_to_blue(self, x_red, y_red, x_blue, y_blue):
        """Determine whether the red agent is close to the blue agent.
           This is the case when the distance between the agents is smaller
           than 10% of the diagonal of the stage.

        Args:
            x_red  = [float] the x pixel coordinate of the red agent
            y_red  = [float] the y pixel coordinate of the red agent
            x_blue = [float] the x pixel coordinate of the blue agent
            y_blue = [float] the y pixel coordinate of the blue agent

        Returns [float]:
            Whether the red agent is close to the blue agent.
        """
        dist_from_blue = self.distance(x_red, y_red, x_blue, y_blue)

        return dist_from_blue < self.diagonal / 10

    def danger_index(self, x_red, y_red, x_blue, y_blue):
        """Compute the 'danger index'. This index is only on when the red
           agent is further from the center than and close to the blue agent.
           The positive values it then returns increase with the distance
           from the center of the stage.

        Args:
            x_red  = [float] the x pixel coordinate of the red agent
            y_red  = [float] the y pixel coordinate of the red agent
            x_blue = [float] the x pixel coordinate of the blue agent
            y_blue = [float] the y pixel coordinate of the blue agent

        Returns [float]:
            The danger index.
        """
        dist = self.dist_from_center(x_red, y_red)
        further = self.further_from_center(x_red, y_red, x_blue, y_blue)
        close = self.close_to_blue(x_red, y_red, x_blue, y_blue)

        return dist * further * close

    def opportunity_index(self, x_red, y_red, x_blue, y_blue):
        """Compute the 'opportunity index'. This index is only on when the red
           agent is closer to the center than and close to the blue agent.
           The positive values it then returns increase with the distance
           from the center of the stage.

        Args:
            x_red  = [float] the x pixel coordinate of the red agent
            y_red  = [float] the y pixel coordinate of the red agent
            x_blue = [float] the x pixel coordinate of the blue agent
            y_blue = [float] the y pixel coordinate of the blue agent

        Returns [float]:
            The opportunity index.
        """
        dist = self.dist_from_center(x_red, y_red)
        further = self.further_from_center(x_red, y_red, x_blue, y_blue)
        close = self.close_to_blue(x_red, y_red, x_blue, y_blue)

        return dist * (1 - further) * close

    def safe_index(self, x_red, y_red):
        """Compute the 'safe index'. This index is always on and increases as
           the red agent moves toward the center.

        Args:
            x_red  = [float] the x pixel coordinate of the red agent
            y_red  = [float] the y pixel coordinate of the red agent

        Returns [float]:
            The safe index.
        """
        max_dist_from_center = self.diagonal / 2
        return max_dist_from_center / self.dist_from_center(x_red, y_red)

    def aggregate(self, xy1_red, xy2_red, xy1_blue, xy2_blue):
        """Aggregate all the relevant metrics in a single vector of numbers.
           The vector will be a torch.Tensor for easy use by PyTorch.

        Args:
            xy1_red  = [(float, float)] x and y pixel coordinates of first
                                        location of red agent
            xy2_red  = [(float, float)] x and y pixel coordinates of second
                                        location of red agent
            xy1_blue  = [(float, float)] x and y pixel coordinates of first
                                        location of blue agent
            xy2_blue  = [(float, float)] x and y pixel coordinates of second
                                        location of blue agent

        Returns [torch.Tensor]:
            A torch.Tensor with all relevant values on the specified device.
        """
        x1_red, y1_red = xy1_red
        x2_red, y2_red = xy2_red
        x1_blue, y1_blue = xy1_blue
        x2_blue, y2_blue = xy2_blue

        l = [x2_red, y2_red, x2_blue, y2_blue,
             self.direction(x1_red, y1_red, x2_red, y2_red),
             self.direction(x1_blue, y1_blue, x2_blue, y2_blue),
             self.speed(x1_red, y1_red, x2_red, y2_red),
             self.speed(x1_blue, y1_blue, x2_blue, y2_blue),
             self.dist_from_center(x2_red, y2_red),
             self.dist_from_center(x2_blue, y2_blue),
             self.dist_from_edge(x2_red, y2_red),
             self.dist_from_edge(x2_blue, y2_blue),
             self.distance(x2_red, y2_red, x2_blue, y2_blue),
             self.direction(x2_red, y2_red, x2_blue, y2_blue),
             self.close_to_blue(x2_red, y2_red, x2_blue, y2_blue),
             self.danger_index(x2_red, y2_red, x2_blue, y2_blue),
             self.opportunity_index(x2_red, y2_red, x2_blue, y2_blue),
             self.safe_index(x2_red, y2_red)
             ]

        return torch.tensor(l, device=self.device)
