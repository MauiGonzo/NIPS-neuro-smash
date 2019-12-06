import math
import torch


class Aggregator(object):
    """Class that is used to compute the input for the agents."""

    def __init__(self, size, time_delta=0.2):
        """Initializes the object with the meta parameters.

        :param size: the width and height, in pixels, of the game
        :param time_delta: the time between location updates
        """
        self.size = size
        self.time_delta = time_delta

        self.diagonal = math.hypot(size, size)
        self.center = size/2, size/2

        self.n_obs = 14  # number of elements in aggregate vector

    def distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Compute the Euclidean distance, in pixels,
           between locations 1 and 2.

        :param x1: the x pixel coordinate of the first location
        :param y1: the y pixel coordinate of the first location
        :param x2: the x pixel coordinate of the second location
        :param y2: the y pixel coordinate of the second location
        :return: The distance between locations 1 and 2.
        """
        return math.hypot(x1-x2, y1-y2)

    def direction(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Compute the direction of movement of the agent, based on
           two consecutive locations, as the counterclockwise angle
           between the vector from (x, y)=(0, 0) to (x, y)=(0, 1)
           and the direction vector in the range (-pi, pi].

        :param x1: the x pixel coordinate of the first location
        :param y1: the y pixel coordinate of the first location
        :param x2: the x pixel coordinate of the second location
        :param y2: the y pixel coordinate of the second location
        :return: The direction the agent is heading in (-pi, pi].
        """
        rad = math.atan2(y1-y2, x1-x2)
        if rad <= -math.pi*1/2:  # rad <= -90 degrees
            return rad + math.pi*3/2  # rad + 270 degrees
        else:
            return rad - math.pi*1/2  # rad - 90 degrees

    def speed(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Compute the speed in the interval between locations 1 and 2
           in pixels per second.

        :param x1: the x pixel coordinate of the first location
        :param y1: the y pixel coordinate of the first location
        :param x2: the x pixel coordinate of the second location
        :param y2: the y pixel coordinate of the second location
        :return: The speed of the agent.
        """
        return self.distance(x1, y1, x2, y2) / self.time_delta

    def dist_from_center(self, x: int, y: int) -> float:
        """Compute the Euclidean distance, in pixels between the location
           and the center of the stage.

        :param x: the x pixel coordinate of the location
        :param y: the y pixel coordinate of the location
        :return: The distance between the location and the center of the stage.
        """
        return self.distance(x, y, self.center[0], self.center[1])

    def further_from_center(self, x_red: int, y_red: int,
                                  x_blue: int, y_blue: int) -> bool:
        """Determine whether the red agent is further away from the center
           of the stage than the blue agent.

        :param x_red: the x pixel coordinate of the location of the red agent
        :param y_red: the y pixel coordinate of the location of the red agent
        :param x_blue: the x pixel coordinate of the location of the red agent
        :param y_blue: the y pixel coordinate of the location of the blue agent
        :return: Whether the red agent is further away from the center than
                 the blue agent.
        """
        dist_from_center_red = self.dist_from_center(x_red, y_red)
        dist_from_center_blue = self.dist_from_center(x_blue, y_blue)

        return dist_from_center_red > dist_from_center_blue

    def close_to_blue(self, x_red: int, y_red: int,
                            x_blue: int, y_blue: int) -> bool:
        """Determine whether the red agent is close to the blue agent.
           This is the case when the distance between the agents is smaller
           than 10% of the diagonal of the stage.

        :param x_red: the x pixel coordinate of the location of the red agent
        :param y_red: the y pixel coordinate of the location of the red agent
        :param x_blue: the x pixel coordinate of the location of the red agent
        :param y_blue: the y pixel coordinate of the location of the blue agent
        :return: Whether the red agent is close to the blue agent.
        """
        dist_from_blue = self.distance(x_red, y_red, x_blue, y_blue)

        return dist_from_blue < self.diagonal / 10

    def danger_index(self, x_red: int, y_red: int,
                           x_blue: int, y_blue: int) -> float:
        """Compute the 'danger index'. This index is only on when the red
           agent is further from the center than and close to the blue agent.
           The positive values it then returns increase with the distance
           from the center of the stage.

        :param x_red: the x pixel coordinate of the location of the red agent
        :param y_red: the y pixel coordinate of the location of the red agent
        :param x_blue: the x pixel coordinate of the location of the red agent
        :param y_blue: the y pixel coordinate of the location of the blue agent
        :return: The danger index.
        """
        dist = self.dist_from_center(x_red, y_red)
        further = self.further_from_center(x_red, y_red, x_blue, y_blue)
        close = self.close_to_blue(x_red, y_red, x_blue, y_blue)

        return dist * further * close

    def opportunity_index(self, x_red: int, y_red: int,
                                x_blue: int, y_blue: int) -> float:
        """Compute the 'opportunity index'. This index is only on when the red
           agent is closer to the center than and close to the blue agent.
           The positive values it then returns increase with the distance
           from the center of the stage.

        :param x_red: the x pixel coordinate of the location of the red agent
        :param y_red: the y pixel coordinate of the location of the red agent
        :param x_blue: the x pixel coordinate of the location of the red agent
        :param y_blue: the y pixel coordinate of the location of the blue agent
        :return: The opportunity index.
        """
        dist = self.dist_from_center(x_red, y_red)
        further = self.further_from_center(x_red, y_red, x_blue, y_blue)
        close = self.close_to_blue(x_red, y_red, x_blue, y_blue)

        return dist * (1 - further) * close

    def safe_index(self, x_red: int, y_red: int) -> float:
        """Compute the 'safe index'. This index is always on and increases as
           the red agent moves toward the center.

        :param x_red: the x pixel coordinate of the location of the red agent
        :param y_red: the y pixel coordinate of the location of the red agent
        :return: The safe index.
        """
        max_dist_from_center = self.diagonal / 2
        return max_dist_from_center / self.dist_from_center(x_red, y_red)

    def aggregate(self, x1_red: int, y1_red: int,
                        x2_red: int, y2_red: int,
                        x1_blue: int, y1_blue: int,
                        x2_blue: int, y2_blue: int) -> torch.Tensor:
        """Aggregate all the relevant metrics in a single vector of numbers.
           The vector will be a torch.Tensor for easy use by PyTorch.

        :param x1_red: x pixel coordinate of first location of red agent
        :param y1_red: y pixel coordinate of first location of red agent
        :param x2_red: x pixel coordinate of second location of red agent
        :param y2_red: y pixel coordinate of second location of red agent
        :param x1_blue: x pixel coordinate of first location of blue agent
        :param y1_blue: y pixel coordinate of first location of blue agent
        :param x2_blue: x pixel coordinate of second location of blue agent
        :param y2_blue: y pixel coordinate of second location of blue agent
        :return: A vector with all relevant values.
        """
        l = [x2_red, y2_red, x2_blue, y2_blue,
             self.direction(x1_red, y1_red, x2_red, y2_red),
             self.direction(x1_blue, y1_blue, x2_blue, y2_blue),
             self.speed(x1_red, y1_red, x2_red, y2_red),
             self.speed(x1_blue, y1_blue, x2_blue, y2_blue),
             self.dist_from_center(x2_red, y2_red),
             self.dist_from_center(x2_blue, y2_blue),
             self.close_to_blue(x2_red, y2_red, x2_blue, y2_blue),
             self.danger_index(x2_red, y2_red, x2_blue, y2_blue),
             self.opportunity_index(x2_red, y2_red, x2_blue, y2_blue),
             self.safe_index(x2_red, y2_red)
             ]

        return torch.tensor(l)
