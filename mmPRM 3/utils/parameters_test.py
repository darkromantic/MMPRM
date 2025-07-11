
def get_parameters():
    params = {
        'num_samples': 3000,
        'ground_ratio': 0.4,
        'R_max': 4,
        'wg': 1,
        'wf': 5,
        'e_factor': 0.1,
        'start': (-24, -5, 0),
        'end': (15.5, 6.5, 11),
        'expand_obstacles': [
            # 斜面
            {'vertices': [[6, 15, 8], [6, 0, 8], [27, 0, 14], [27, 15, 14],
                         [6, 15, 0], [6, 0, 0], [27, 0, 0], [27, 15, 0]],
             'faces': [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6],
                       [0,3,7,4], [1,2,6,5]]},

            # 障碍物1
            {'vertices': [[-8, 15, 8], [-8, -7, 8], [2, -7, 8], [2, 15, 8],
                            [-8, 15, 0], [-8, -7, 0], [2, -7, 0], [2, 15, 0]],
             'faces': [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6],
                       [0,3,7,4], [1,2,6,5]]},

            # 障碍物2
            {'vertices': [[-14, -15, 8], [-14, 5, 8], [-8, 5, 8], [-8, -15, 8],
                            [-14, -15, 0], [-14, 5, 0], [-8, 5, 0], [-8, -15, 0]],
             'faces': [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6],
                       [0,3,7,4], [1,2,6,5]]},
        ],
        'connect_obstacles': [
            # 斜面
            {'vertices': [[4, 15, 8], [4, -2, 8], [27, -2, 14], [27, 15, 14],
                         [4, 15, 0], [4, -2, 0], [27, -2, 0], [27, 15, 0]],
             'faces': [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6],
                       [0,3,7,4], [1,2,6,5]]},

            # 障碍物1
            {'vertices': [[-5, 15, 8], [-5, -5, 8], [-1, -5, 8], [-1, 15, 8],
                            [-5, 15, 0], [-5, -5, 0], [-1, -5, 0], [-1, 15, 0]],
             'faces': [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6],
                       [0,3,7,4], [1,2,6,5]]},

            # 障碍物2
            {'vertices': [[-12, -15, 8], [-12, 3, 8], [-10, 3, 8], [-10, -15, 8],
                            [-12, -15, 0], [-12, 3, 0], [-10, 3, 0], [-10, -15, 0]],
             'faces': [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6],
                       [0,3,7,4], [1,2,6,5]]},
        ],
        'real_obstacles': [
            # 斜面
            {'vertices': [[4, 15, 8], [4, -2, 8], [27, -2, 14], [27, 15, 14],
                         [4, 15, 0], [4, -2, 0], [27, -2, 0], [27, 15, 0]],
             'faces': [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6],
                       [0,3,7,4], [1,2,6,5]]},

            # 障碍物1
            {'vertices': [[-6, 15, 8], [-6, -5, 8], [0, -5, 8], [0, 15, 8],
                            [-6, 15, 0], [-6, -5, 0], [0, -5, 0], [0, 15, 0]],
             'faces': [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6],
                       [0,3,7,4], [1,2,6,5]]},

            # 障碍物2
            {'vertices': [[-12, -15, 8], [-12, 3, 8], [-10, 3, 8], [-10, -15, 8],
                            [-12, -15, 0], [-12, 3, 0], [-10, 3, 0], [-10, -15, 0]],
             'faces': [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6],
                       [0,3,7,4], [1,2,6,5]]},
        ],
        'xlim': (-27, 27),
        'ylim': (-15, 15),
        'zlim': (0, 18),
    }
    return params