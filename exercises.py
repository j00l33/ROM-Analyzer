class exercise:
    def __init__(self, start_angles, measure_angles, measure_distances, code, primary_angle, primary_angle_idx = 0, primary_distance_idx = 0, ):
        self.start_angles = start_angles
        self.measure_angles = measure_angles
        self.measure_distances = measure_distances
        self.code = code
        self.primary_angle = primary_angle
        self.primary_angle_idx = primary_angle_idx
        self.primary_distance_idx = primary_distance_idx

movements = []

SeatedLegCurl = exercise([[[5, 11, 13], [60, 120]], [[11, 13, 15], [135, 200]]], [[11, 13, 15]], [[11,15]], "SLC", True, 0)
movements.append(SeatedLegCurl)