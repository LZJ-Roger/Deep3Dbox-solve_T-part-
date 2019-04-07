import numpy as np
import math
import itertools
#from sympy import *

def solve_T(points_2D, RoarNet2D_pred):
    '''
    :param points_2D: numpy array, POINTS ON THE IMAGE
    :param points_3D: POINTS ON THE WORLD-COR, 8x3
    :param RoarNet2D_pred: numpy array, PREDICTION FROM 2DNet
    :return: global_T: numpy array 得出的是相机系原点到物体中心的向量
    '''
    global_T = []
    global_iou = -1
    global_norm = 100000000000

    h = RoarNet2D_pred[0]
    w = RoarNet2D_pred[1]
    l = RoarNet2D_pred[2]
    theta = RoarNet2D_pred[3]

    if theta > np.pi:
        theta = theta - 2 * np.pi
    if theta < -np.pi:
        theta = theta + 2 * np.pi

    theta = -(np.pi / 2 - theta)  # no -

    corner = np.array([[w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                       [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
                       [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2]])

    points_3D = corner.transpose()

    corner2 = list(corner)
    corner2.append([1, 1, 1, 1, 1, 1, 1, 1])
    corner2 = np.array(corner2)

    R = [[math.cos(theta), 0, math.sin(theta)],
         [0, 1, 0],
         [-math.sin(theta), 0, math.cos(theta)]]
    R = np.array(R)

    K = [[719.787081,    0.,            608.463003],
         [0.,            719.787081,    174.545111],
         [0.,            0.,            1.]]
    K = np.array(K)

    # character = list(itertools.permutations(points_3D, 4))  # permutations
    for i1 in [0, 1, 2, 3]:
     for i2 in [0, 1, 2, 3]:
      for i3 in [4, 5, 6, 7]:
       for i4 in [4, 5, 6, 7]:
        p_3D = []
        p_3D.append(points_3D[i1, :])
        p_3D.append(points_3D[i2, :])
        p_3D.append(points_3D[i3, :])
        p_3D.append(points_3D[i4, :])
    #for i in range(3600):  # len(character)
        # print(i1)
        # p_3D = np.array(character[i])
        '''
        p_3D = []
        for k in range(4):
            idx = np.random.choice(range(8))
            p_3D.append(points_3D[idx, :])
        '''
        p_3D = np.array(p_3D)

        # p_3D = np.array(character[i])
        # t1 = Symbol('t1')
        # t2 = Symbol('t2')
        # t3 = Symbol('t3')

        # func = []

        Matr = []
        B = []

        tmp1 = np.matmul(R, p_3D[0, :].T)
        A1 = [[1, 0, 0, tmp1[0]],
              [0, 1, 0, tmp1[1]],
              [0, 0, 1, tmp1[2]]]
        A1 = np.array(A1)
        A1 = np.dot(K, A1)
        Matr.append([A1[0, 0]-A1[2, 0]*points_2D[0], A1[0, 1]-A1[2, 1]*points_2D[0], A1[0, 2]-A1[2, 2]*points_2D[0]])
        B.append(-A1[0, 3]+A1[2, 3]*points_2D[0])
        #func.append((A1[0, 0]*t1+A1[0, 1]*t2+A1[0, 2]*t3+A1[0, 3])
        #            -(A1[2, 0]*t1+A1[2, 1]*t2+A1[2, 2]*t3+A1[2, 3])*points_2D[0])

        tmp2 = np.matmul(R, p_3D[1, :].T)
        A2 = [[1, 0, 0, tmp2[0]],
              [0, 1, 0, tmp2[1]],
              [0, 0, 1, tmp2[2]]]
        A2 = np.array(A2)
        A2 = np.dot(K, A2)
        Matr.append([A2[1, 0] - A2[2, 0] * points_2D[1], A2[1, 1] - A2[2, 1] * points_2D[1], A2[1, 2] - A2[2, 2] * points_2D[1]])
        B.append(-A2[1, 3] + A2[2, 3] * points_2D[1])
        #func.append((A2[1, 0] * t1 + A2[1, 1] * t2 + A2[1, 2] * t3 + A2[1, 3])
        #            -(A2[2, 0] * t1 + A2[2, 1] * t2 + A2[2, 2] * t3 + A2[2, 3])*points_2D[1])

        tmp3 = np.matmul(R, p_3D[2, :].T)
        A3 = [[1, 0, 0, tmp3[0]],
              [0, 1, 0, tmp3[1]],
              [0, 0, 1, tmp3[2]]]
        A3 = np.array(A3)
        A3 = np.dot(K, A3)
        Matr.append([A3[0, 0] - A3[2, 0] * points_2D[2], A3[0, 1] - A3[2, 1] * points_2D[2], A3[0, 2] - A3[2, 2] * points_2D[2]])
        B.append(-A3[0, 3] + A3[2, 3] * points_2D[2])
        #func.append((A3[0, 0] * t1 + A3[0, 1] * t2 + A3[0, 2] * t3 + A3[0, 3])
        #            -(A3[2, 0] * t1 + A3[2, 1] * t2 + A3[2, 2] * t3 + A3[2, 3])*points_2D[2])

        tmp4 = np.matmul(R, p_3D[3, :].T)
        A4 = [[1, 0, 0, tmp4[0]],
              [0, 1, 0, tmp4[1]],
              [0, 0, 1, tmp4[2]]]
        A4 = np.array(A4)
        A4 = np.dot(K, A4)
        Matr.append([A4[1, 0] - A4[2, 0] * points_2D[3], A4[1, 1] - A4[2, 1] * points_2D[3], A4[1, 2] - A4[2, 2] * points_2D[3]])
        B.append(-A4[1, 3] + A4[2, 3] * points_2D[3])
        #func.append((A4[1, 0] * t1 + A4[1, 1] * t2 + A4[1, 2] * t3 + A4[1, 3])
        #            -(A4[2, 0] * t1 + A4[2, 1] * t2 + A4[2, 2] * t3 + A4[2, 3])*points_2D[3])


        Matr = np.array(Matr)
        B = np.array(B)
        B = np.expand_dims(B, 1)

        # T = np.matmul(np.linalg.pinv(Matr), B)
        T, norm, _, _ = np.linalg.lstsq(Matr, B)

        if (T[2, 0]<=0):
            continue


        M1 = np.hstack((R, T))
        # P = np.matmul(K, M1)

        # ------------------------------------ #

        box3donimg = np.dot(M1, corner2)
        box3donimg = np.dot(K, box3donimg)
        # box3donimg = box3donimg.transpose()
        box3donimg[0] /= (box3donimg[2] + np.finfo(np.float32).eps)
        box3donimg[1] /= (box3donimg[2] + np.finfo(np.float32).eps)

        F = True
        imgwid = (points_2D[2] - points_2D[0])
        imghei = (points_2D[3] - points_2D[1])
        for iii in range(8):
            if (box3donimg[0, iii] < points_2D[0] - 0.2 * imgwid or box3donimg[0, iii] > points_2D[2] + 0.2 * imgwid):
                F = False
            if (box3donimg[1, iii] < points_2D[1] - 0.2 * imghei or box3donimg[1, iii] > points_2D[3] + 0.2 * imghei):
                F = False

        if (norm[0] < global_norm and F == True):
            global_T = T
            global_norm = norm[0]

        '''
        xmax = np.max(box3donimg[0, :])
        xmin = np.min(box3donimg[0, :])
        ymax = np.max(box3donimg[1, :])
        ymin = np.min(box3donimg[1, :])

        if(xmin<0 or ymin<0 or xmax>1224 or ymax>370):
            continue

        rec = np.array([ymin, xmin, ymax, xmax])
        iou = cal_IOU(np.array([points_2D[1], points_2D[0], points_2D[3], points_2D[2]]), rec)

        if(iou>global_iou):
            global_T = T
            global_iou = iou
            # print('success ! ')
        '''

    return np.array(global_T), global_norm


def cal_IOU(rec1, rec2):
    # computing area of each rectangles
    ttt = rec2[2] - rec2[0]
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


if __name__ == '__main__':
    points_2D = np.array([390.79803, 164.44536, 450.4052, 198.81921])
    RoarNet2D_pred = [3.6077675533691407, 2.7363950778683472, 35.063437587495116, 1.6609269701170407]
    T, _ = solve_T(points_2D, RoarNet2D_pred)
    T[1, 0] = T[1, 0] + 3.6077675533691407/2
    print(T)
