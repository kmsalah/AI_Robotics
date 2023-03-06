"""
- All the estimates should be given relative to your robot's starting location, which can be represented as (0,0).

- Measurements: Your robot will receive measurements from gems located throughout the terrain in the format:
           {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'D'}, ...}.

- Only gems that have not been collected and are within the horizon distance will return measurements.

- Movements: Your robot's movements are stochastic and are subject to constraints. If the maximum distance or steering is exceeded, the robot will not move. 
The robot's movements are executed with an action: 'move 1.570963 1.0', which will turn the robot counterclockwise 90 degrees and then move 1.0.

- Needed Gems: You will receive a list of gem types that your robot needs to extract from the environment. These gems are represented by an uppercase letter of the alphabet (ABC...). You can extract a gem from the current location with the action 'extract', but your robot must be within 0.15 distance to successfully extract it. Once a gem is extracted, it will no longer exist in the environment and won't return a measurement. There may be gems in the environment that are not required to be extracted.
- The robot will always execute a measurement first, followed by an action.
- The robot will have a time limit of 5 seconds to find and extract all the needed gems.
"""
import copy
import math
from typing import Dict, List
from matrix import matrix
from robot import Robot, truncate_angle, compute_distance, compute_bearing, PI

class SLAM:
    """Create a basic SLAM module.
    """
    def __init__(self):
        """Initialize SLAM components here.
        """
        # important vars
        self.Omega = matrix()
        self.Xi = matrix()
        self.seen_landmarks = {}
        self.dim = 0
        self.bearing = 0  # starting bearing is 0 per project notes
        self.mu = matrix()

        # initialize matrices
        n = 2  # number of seen landmarks + 1 * 2
        self.Omega.zero(n, n)  # [[0,0],[0,0]]
        self.Omega.value[0][0] = 1.0  # [[1,0],[0,0]]
        self.Omega.value[1][1] = 1.0  # [[1,0],[0,1]]

        self.Xi.zero(n, 1)  # [[0][0]]

    def get_points_to_plot(self):
        """
               Retrieves the x, y locations for all landmarks

               Args:
                   None

               Returns:
                   all the landmark coordinates relative to the robots frame with an initial position of 0.0
        """
        res = {}
        for id in self.seen_landmarks:
            x, y = self.get_coordinates_by_landmark_id(id)
            point = (x, y)
            res[id] = point
        return res

    # Provided Functions
    def get_coordinates_by_landmark_id(self, landmark_id: str):
        """
        Retrieves the x, y locations for a given landmark

        Args:
            landmark_id: The id for a processed landmark

        Returns:
            the coordinates relative to the robots frame with an initial position of 0.0
        """
        n = self.seen_landmarks[landmark_id]

        x, y = self.mu.value[n][0], self.mu.value[n + 1][0]
        return x, y

    def process_measurements(self, measurements: Dict):
        """
        Process a new series of measurements.

        Args:
            measurements: Collection of measurements
                in the format {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'B'}, ...}

        Returns:
            x, y: current belief in location of the robot
        """
        for landmark_id in measurements:
            measurement = measurements[landmark_id]
            distance = measurement['distance']
            bearing = measurement['bearing']

            measurement_noise = [1.0, 1.0]

            dx = distance * math.cos(bearing)
            dy = distance * math.sin(bearing)

            odom = [dx, dy]  # distance to landmark

            if landmark_id not in self.seen_landmarks:
                '''
                self.seen_landmarks[landmark_id] = self.num_landmarks

                dim = 2 * (1 + self.num_landmarks)
                matrix_index = [r for r in range(4, dim + 2)]
                self.Omega = self.Omega.expand(dim + 2, dim + 2, matrix_index,
                                               matrix_index)
                self.Xi = self.Xi.expand(dim + 2, 1, matrix_index, [0])
                self.num_landmarks += 1
                '''
                dim = 2 * (1 + len(self.seen_landmarks))
                self.Xi = self.Xi.expand(dim + 2, 1, [r for r in range(dim)], [0])
                self.Omega = self.Omega.expand(dim + 2, dim + 2, [r for r in range(dim)], [c for c in range(dim)])
                self.seen_landmarks[landmark_id] = dim

            i = self.seen_landmarks[landmark_id]
            for b in range(2):
                self.Omega.value[b][b] += 1.0 / measurement_noise[b]
                self.Omega.value[i + b][i + b] += 1.0 / measurement_noise[b]
                self.Omega.value[b][i + b] += -1.0 / measurement_noise[b]
                self.Omega.value[i + b][b] += -1.0 / measurement_noise[b]
                self.Xi.value[b][0] += -odom[b] / measurement_noise[b]
                self.Xi.value[i + b][0] += odom[b] / measurement_noise[b]

        mu = self.Omega.inverse() * self.Xi
        self.mu = mu
        x = mu.value[0][0]
        y = mu.value[1][0]
        return x, y

    def process_movement(self, steering: float, distance: float):
        """
        Process a new movement.

        Args:
            steering: amount to turn
            distance: distance to move

        Returns:
            x, y: current belief in location of the robot
        """

        self.bearing = truncate_angle(steering + self.bearing)
        dx = distance * math.cos(self.bearing)
        dy = distance * math.sin(self.bearing)
        motion = [dx, dy]

        cols = len(self.Omega.value[0])
        dim = 2 * (1 + len(self.seen_landmarks))
        aList = [0, 1] + [r for r in range(4, dim + 2)]
        self.Omega = self.Omega.expand(dim + 2, cols + 2, aList,
                                       aList)
        self.Xi = self.Xi.expand(dim + 2, 1, aList, [0])

        motion_noise = 1.0

        for b in range(4):
            self.Omega.value[b][b] += 1.0 / motion_noise
        for b in range(2):
            self.Omega.value[b][2 + b] += -1.0 / motion_noise
            self.Omega.value[b + 2][b] += -1.0 / motion_noise
            self.Xi.value[b][0] += -motion[b] / motion_noise
            self.Xi.value[b + 2][0] += motion[b] / motion_noise

        bList = list(range(2, dim + 2))
        A = self.Omega.take([0, 1], bList)
        B = self.Omega.take([0, 1])
        C = self.Xi.take([0, 1], [0])
        self.Xi = self.Xi.take(bList, [0]) - A.transpose() * B.inverse() * C
        self.Omega = self.Omega.take(bList, bList) - A.transpose() * B.inverse() * A

        mu = self.Omega.inverse() * self.Xi
        self.mu = mu
        x = mu.value[0][0]
        y = mu.value[1][0]
        return x, y


class ActionPlanner:
    """
    Create a planner to navigate the robot to reach and extract all the needed gems from an unknown start position.
    """

    def __init__(self, max_distance, max_steering):
        """Initialize your planner here.
        Args:
            max_distance(float): the max distance the robot can travel in a single move.
            max_steering(float): the max steering angle the robot can turn in a single move.
        """

        self.slam = SLAM()
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.x = 0
        self.y = 0
        self.bearing = 0
        self.visited_in_search = set()
        self.remaining_in_search = set()
        self.measured_landmarks = set()
        self.seen_gem_types = set()
        self.landmarks_ids_to_gem_types = {}
        self.in_middle_of_search = False

    def next_move(self, needed_gems: List[str], measurements: Dict):
        """Next move based on the current set of measurements.
        Args:
            needed_gems: List of gems remaining which still need to be found and extracted.
            measurements: Collection of measurements from gems in the area.
                                {'landmark id': {
                                                    'distance': 0.0,
                                                    'bearing' : 0.0,
                                                    'type'    :'B'
                                                },
                                ...}
        Return: action: str, points_to_plot: dict [optional]
            action (str): next command to execute on the robot.
                allowed:
                    'move 1.570963 1.0'  - Turn left 90 degrees and move 1.0 distance.
                    'extract B 1.5 -0.2' - [Part B] Attempt to extract a gem of type B from your current location.
                                           This will succeed if the specified gem is within the minimum sample distance.
            points_to_plot (dict): point estimates (x,y) to visualize if using the visualization tool [optional]
                            'self' represents the robot estimated position
                            <landmark_id> represents the estimated position for a certain landmark
                format:
                    {
                        'self': (x, y),
                        '<landmark_id_1>': (x1, y1),
                        '<landmark_id_2>': (x2, y2),
                        ....
                    }
        """
        self.x, self.y = self.slam.process_measurements(measurements)
        self.bearing = self.slam.bearing
        #print("bearing")
        #print(self.bearing)
        #print("needed gems:")
        #print(needed_gems)

        #print("measurements")
        #for i in measurements:
            #print(measurements[i]['type'])

        for landmark_id in measurements:
            if landmark_id not in self.measured_landmarks:
                self.measured_landmarks.add(landmark_id)
                self.seen_gem_types.add(measurements[landmark_id]['type'])
                self.landmarks_ids_to_gem_types[landmark_id] = measurements[landmark_id]['type']

        for gem in needed_gems:
            if gem in self.seen_gem_types:
                #print("target")
                #print(gem)
                self.visited_in_search.clear()
                self.remaining_in_search.clear()
                self.in_middle_of_search = False
                gem_to_visit_id = None

                for landmark_id in self.measured_landmarks:
                    gem_type = self.landmarks_ids_to_gem_types[landmark_id]
                    if gem == gem_type:
                        gem_to_visit_id = landmark_id

                gem_x, gem_y = self.slam.get_coordinates_by_landmark_id(gem_to_visit_id)
                point = (gem_x, gem_y)
                #print(point)
                current_position = (self.x, self.y)
                #print(current_position)


                distance_to_gem = compute_distance(current_position, point)
                bearing_to_gem = truncate_angle(compute_bearing(current_position, point) - self.bearing)
                #print("bearing to gem")
                #print(bearing_to_gem)

                #if we're already in the same direction, set the bearing to gem to 0
                if abs(bearing_to_gem - self.bearing) < .10: #.10 is just a rough magic number for now
                    #print("hit")
                    bearing_to_gem = 0

                #print("-")
                type_of_gem = self.landmarks_ids_to_gem_types[gem_to_visit_id]

                if distance_to_gem > 0.15:
                    steering = max(-self.max_steering, bearing_to_gem)
                    steering = min(self.max_steering, steering)
                    distance = max(0, distance_to_gem)
                    distance = min(self.max_distance, distance)
                    self.x, self.y = self.slam.process_movement(steering, distance)
                    action = 'move ' + str(steering) + ' ' + str(distance)
                    #print(action)
                    return action, {}
                elif distance_to_gem <= 0.15:
                    action = 'extract ' + str(type_of_gem) + ' ' + str(gem_x) + ' ' + str(gem_y)
                    #print(action)
                    return action, {}

        # we have not seen gem
        # drive around to different landmarks we've seen
        if not self.in_middle_of_search:
            #print("starting search")
            self.remaining_in_search = copy.deepcopy(self.measured_landmarks)
            self.in_middle_of_search = True
            #print("reset search")
            if len(self.remaining_in_search) == 0:
                self.x, self.y = self.slam.process_movement(self.max_steering, self.max_distance)
                action = 'move ' + str(self.max_steering) + ' ' + str(self.max_distance)
                self.in_middle_of_search = True
                return action, {}


        else:
            self.remaining_in_search = self.measured_landmarks.difference(self.visited_in_search)

            if len(self.remaining_in_search) == 0:
                #couldn't find anything new after going to all known landmarks
                origin_x, origin_y = 0, 0
                point = (origin_x, origin_y)
                current_position = (self.x, self.y)
                distance_to_gem = compute_distance(current_position, point)
                bearing_to_gem = truncate_angle(compute_bearing(current_position, point) - self.bearing)
                steering = max(-self.max_steering, bearing_to_gem)
                steering = min(self.max_steering, steering)
                distance = max(0, distance_to_gem)
                distance = min(self.max_distance, distance)
                self.x, self.y = self.slam.process_movement(steering, distance)
                action = 'move ' + str(steering) + ' ' + str(distance)
                self.in_middle_of_search = False
                return action, {}

            #print("seen landmarks")
            #print(self.measured_landmarks)

            #print("visited")
            #print(self.visited_in_search)

            #print("remaining")
            #print(self.remaining_in_search)

        # get first gem we could visit
        gem_to_visit_id = list(self.remaining_in_search)[0]
        #print("current target")
        #print(self.landmarks_ids_to_gem_types[gem_to_visit_id])

        gem_x, gem_y = self.slam.get_coordinates_by_landmark_id(gem_to_visit_id)
        point = (gem_x, gem_y)
        current_position = (self.x, self.y)
        distance_to_gem = compute_distance(current_position, point)
        bearing_to_gem = truncate_angle(compute_bearing(current_position, point) - self.bearing)

        if distance_to_gem > 0.15:
            steering = max(-self.max_steering, bearing_to_gem)
            steering = min(self.max_steering, steering)
            distance = max(0, distance_to_gem)
            distance = min(self.max_distance, distance)
            self.x, self.y = self.slam.process_movement(steering, distance)
            action = 'move ' + str(steering) + ' ' + str(distance)
            #print(action)
            return action, {}
        elif distance_to_gem <= 0.15:
            self.visited_in_search.add(gem_to_visit_id)
            action = 'move ' + str(0) + ' ' + str(0)
            #print(action)
            return action, {}

