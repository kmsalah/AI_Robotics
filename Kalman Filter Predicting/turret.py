from builtins import object
import numpy as np

from matrix import matrix

H = matrix([[1., 0, 0, 0, 0, 0],
            [0, 1., 0, 0, 0, 0]])

identity = matrix([[1., 0, 0, 0, 0, 0],
                   [0, 1., 0, 0, 0, 0],
                   [0, 0, 1., 0, 0, 0],
                   [0, 0, 0, 1., 0, 0],
                   [0, 0, 0, 0, 1., 0],
                   [0, 0, 0, 0, 0, 1.]])

F = matrix([[1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
R = matrix([[0.0625, 0], [0, .0625]])


def initalized_p_matrix(sigma_sq):
    o2 = sigma_sq
    return matrix ([[o2, 0, 0, 0, 0, 0],
                   [0, o2, 0, 0, 0, 0],
                   [0, 0, o2, 0, 0, 0],
                   [0, 0, 0, o2, 0, 0],
                   [0, 0, 0, 0, o2, 0],
                   [0, 0, 0, 0, 0, o2]])


def initalized_state_matrix():
    return matrix([[0], [0], [0], [0], [0], [0]])


class Turret(object):

    def __init__(self, init_pos, arena_contains_fcn, max_angle_change,
                 initial_state):
        self.x_pos = init_pos['x']
        self.y_pos = init_pos['y']

        self.bounds_checker = arena_contains_fcn
        self.current_aim_rad = initial_state['h']
        self.max_angle_change = max_angle_change

        self.measurements = {}  # <meteorite_id, (x,y)>
        self.states = {}  # <ids, (matrix(x, xdot, xdotdot), matrix(x, xdot, xdotdot)>
        self.predictions = {}  # <ids, (matrix(x, xdot, xdotdot), matrix(x, xdot, xdotdot)>

        self.P_matrices = {}

    def get_meteorite_observations(self, meteorite_locations):
        """Observe and record the locations of the meteorites. Use the self-reference to indicate the current object, 
           which is the Turret. The meteorite_locations variable is a list that contains observations of the meteorite locations.
           Each observation in the list is represented as a tuple that includes a unique ID for the meteorite (i), and its x and y 
           locations (with some noise) at the current timestep. To identify specific meteorites, use their IDs instead of their positions
           within the list, as the list may change in size when some meteorites move out of bounds. The goal is to store the 
           meteorite_locations data in this function for use in predicting the meteorites' future locations, and also to update the various 
           components of the Kalman Filter to enable the Turret to perform that prediction in the do_kf_estimate_meteorites() function.

        Returns: None
        """
        for m_tpl in meteorite_locations:
            id = m_tpl[0]

            sigma_sq = 1

            if id not in self.predictions:
                self.predictions[id] = initalized_state_matrix()

            if id not in self.P_matrices:
                self.P_matrices[id] = initalized_p_matrix(sigma_sq)

            self.measurements[id] = matrix([[m_tpl[1]], [m_tpl[2]]])


        pass

    def do_kf_estimate_meteorites(self):
        """Estimate the locations of meteorites one timestep in the future.

        Returns: tuple of tuples containing:
            (meteorite ID, meteorite x-coordinate, meteorite y-coordinate)
        """
        tuples = []


        for id in self.predictions:
            # current state matrix
            cur_s = self.predictions[id]
            X = F * cur_s


            P = self.P_matrices[id]
            P = F * P * F.transpose()

            z = self.measurements[id]

            y = z - (H * X)
            S = H*P * H.transpose() + R

            K = P * H.transpose() * S.inverse()

            x = X + (K * y)
            P = (identity - (K * H)) * P

            self.predictions[id] = x
            self.P_matrices[id] = P
            returnable = (id, x[0][0], x[1][0])
            tuples.append(returnable)



        return tuples

