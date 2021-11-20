import numpy as np
import LQR
import matplotlib.pyplot as plt
import itertools
from scipy.stats import norm

time = 50
arr = []

mode = 'none'

if mode is 'none':

    reward = []
    mean_arr_one = []
    mean_arr_two = []
    up_arr_one = []
    up_arr_two = []
    down_arr_one = []
    down_arr_two = []

    for i in range(20):  # get data 50 times
        pos, val = LQR.run_system(time, 'opt')
        arr = np.append(arr, pos)
        reward = np.append(reward, val)

    for i in range(time):
        sone = []
        stwo = []

        for j in range(20):
            sone.append(arr[2 * i + 100 * j])
            stwo.append(arr[2 * i + 1 + 100 * j])

        mean_arr_one.append(np.mean(sone))
        mean_arr_two.append(np.mean(stwo))
        up_arr_one.append(2 * np.std(sone) + np.mean(sone))
        up_arr_two.append(2 * np.std(stwo) + np.mean(stwo))
        down_arr_one.append(np.mean(sone) - 2 * np.std(sone))
        down_arr_two.append(np.mean(stwo) - 2 * np.std(stwo))

    plt.plot(range(1, time + 1), mean_arr_one, 'b')
    plt.plot(range(1, time + 1), mean_arr_two, 'r')
    plt.fill_between(range(1, time + 1), up_arr_two, down_arr_two, color='r')
    plt.fill_between(range(1, time + 1), up_arr_one, down_arr_one, color='b')

    plt.show()

elif mode is 'p':
    arr_zero = []
    mean_arr = []
    mean_arr_zero = []
    up_arr = []
    up_arr_zero = []
    down_arr = []
    down_arr_zero = []

    for i in range(20):  # get data 50 times
        pos, val = LQR.run_system(time, 'p')
        arr = np.append(arr, pos)
        pos, val = LQR.run_system(time, 'p_zero')
        arr_zero = np.append(arr_zero, pos)

    for i in range(time):
        s = []
        s_zero = []

        for j in range(20):
            s.append(arr[2 * i + 100 * j])
            s_zero.append(arr_zero[2 * i + 100 * j])

        mean_arr.append(np.mean(s))
        mean_arr_zero.append(np.mean(s_zero))
        up_arr.append(2 * np.std(s) + np.mean(s))
        up_arr_zero.append(2 * np.std(s_zero) + np.mean(s_zero))
        down_arr.append(np.mean(s) - 2 * np.std(s))
        down_arr_zero.append(np.mean(s_zero) - 2 * np.std(s_zero))

    plt.plot(range(1, time + 1), mean_arr, 'b')
    plt.plot(range(1, time + 1), mean_arr_zero, 'r')
    plt.fill_between(range(1, time + 1), up_arr_zero, down_arr_zero, color='r')
    plt.fill_between(range(1, time + 1), up_arr, down_arr, color='b')
    plt.legend(['P-controller', 'P with zero'])

    plt.show()

else:
    mean = np.zeros((6, 50))
    stand = np.zeros((12, 50))
    controller = ['none', 'p', 'opt']
    k = 1
    for c in controller:
        reward = []
        mean_arr_one = []
        mean_arr_two = []
        up_arr_one = []
        up_arr_two = []
        down_arr_one = []
        down_arr_two = []

        for i in range(20):  # get data 50 times
            pos, val = LQR.run_system(time, c)
            arr = np.append(arr, pos)
            reward = np.append(reward, val)

        for i in range(time):
            sone = []
            stwo = []

            for j in range(20):
                sone.append(arr[2 * i + 100 * j])
                stwo.append(arr[2 * i + 1 + 100 * j])

            mean_arr_one.append(np.mean(sone))
            mean_arr_two.append(np.mean(stwo))
            up_arr_one.append(2 * np.std(sone) + np.mean(sone))
            up_arr_two.append(2 * np.std(stwo) + np.mean(stwo))
            down_arr_one.append(np.mean(sone) - 2 * np.std(sone))
            down_arr_two.append(np.mean(stwo) - 2 * np.std(stwo))

        plt.subplot(3, 1, k)
        # fig.manager.set_window_title('comparison of solutions')
        plt.plot(range(1, time + 1), mean_arr_one, 'b')
        plt.plot(range(1, time + 1), mean_arr_two, 'r')
        plt.fill_between(range(1, time + 1), up_arr_one, down_arr_one, color='b')
        plt.fill_between(range(1, time + 1), up_arr_two, down_arr_two, color='r')
        k += 1



    plt.show()

    input('press enter to close')


