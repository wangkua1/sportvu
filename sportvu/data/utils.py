import numpy as np


def shuffle_2_array(x, y):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    return x, y


def create_circle(radius):
    r_squared = np.arange(-radius, radius + 1)**2
    dist_to = r_squared[:, None] + r_squared
    # ones_circle = (dist_to <= radius**2).astype('float32')
    ones_circle = 1 - dist_to.astype('float32')**2 / ((radius + 1) * 2)**2
    circle_length = radius * 2 + 1
    return ones_circle, int(circle_length)


def pictorialize(xx, sample_rate=1, Y_RANGE=100, X_RANGE=50, radius=3):
    """
    xx of shape (..., 2)
    return: shape (..., Y_RANGE, X_RANGE) one hot encoded pictures
    WARNING: should really try to vectorize ***
    """
    # some preprocessing to make sure data is within range
    xx[:, 0][xx[:, 0] >= Y_RANGE] = Y_RANGE - 1
    xx[:, 1][xx[:, 1] >= X_RANGE] = X_RANGE - 1
    xx[xx <= 0] = 0
    ###
    xx = np.array(xx).astype('int32')
    old_shape = list(xx.shape)
    Y_RANGE = Y_RANGE / sample_rate
    X_RANGE = X_RANGE / sample_rate
    # reasons behind padding radius is to avoid boundary cases when filling
    # with circles
    target = np.zeros(
        (old_shape[:-1] + [int(Y_RANGE + 2 * radius), int(X_RANGE + 2 * radius)]))
    nr_xx = xx.reshape(-1, xx.shape[-1])
    nr_target = target.reshape(-1, target.shape[-2], target.shape[-1])
    # create the small circle first
    ones_circle, circle_length = create_circle(radius)
    circles = ones_circle
    # fill it up
    ind0 = np.arange(nr_target.shape[0]).astype('int32')
    start_x = (nr_xx[:, 0] / sample_rate).astype('int32')
    start_y = (nr_xx[:, 1] / sample_rate).astype('int32')
    for ind in xrange(len(ind0)):  # WARNING ***
        nr_target[ind0[ind], start_x[ind]:start_x[ind] + circle_length,
                  start_y[ind]:start_y[ind] + circle_length] = circles
    if radius > 0:
        nr_target = nr_target[:, radius:-radius,
                              radius:-radius]  # shave off the padding
    target = nr_target.reshape((old_shape[:-1] + [int(Y_RANGE), int(X_RANGE)]))
    return target


def pictorialize_team(xx, sample_rate=1, Y_RANGE=100, X_RANGE=50, radius=0):
    """
    LEGACY FUNCTION
    please use caller_f_team to achieve team processing
    """
    """
    xx of shape (..., 2*N_PLAYERS)
    basically calls pictorialize N_PLAYERS times and combine the results
    """
    rolled_xx = np.rollaxis(xx, -1)
    for sli in xrange(int(rolled_xx.shape[0] / 2)):
        player = rolled_xx[2 * sli:2 * (sli + 1)]
        tmp = pictorialize(np.rollaxis(player, 0, len(
            player.shape)), sample_rate, Y_RANGE, X_RANGE, radius)
        retval = retval + tmp if sli > 0 else tmp
    return retval


def make_3teams_11players(sequence):
    ret = []
    for team in sequence:
        for player in team:
            ret.append(player)
    return ret


def pictorialize_fast(xx, sample_rate=1, Y_RANGE=100, X_RANGE=50):
    """
    xx of shape (Batch, Players=11, Time, 2)
    return: shape (Batch, Teams=3, Time, Y_RANGE, X_RANGE) one hot encoded pictures
    """
    old_shape = list(xx.shape)
    assert (old_shape[1] == 11 and
            old_shape[-1] == 2 and
            len(old_shape) == 4)
    # some preprocessing to make sure data is within range
    xx[:, :, :, 0][xx[:, :, :, 0] >= Y_RANGE] = Y_RANGE - 1
    xx[:, :, :, 1][xx[:, :, :, 1] >= X_RANGE] = X_RANGE - 1
    xx[xx <= 0] = 0
    ###
    xx = np.array(xx).astype('int32')

    Y_RANGE = Y_RANGE / sample_rate
    X_RANGE = X_RANGE / sample_rate
    target_shape = (  # B, P, T, Y, X
        old_shape[0], 11, old_shape[2], Y_RANGE, X_RANGE)
    target = np.zeros(target_shape)

    nr_xx = xx.reshape(-1, xx.shape[-1])
    nr_target = target.reshape(-1, target.shape[-2], target.shape[-1])
    # fill it up
    ind0 = np.arange(nr_target.shape[0]).astype('int32')
    nr_target[ind0, nr_xx[:, 0] / sample_rate, nr_xx[:, 1] / sample_rate] = 1.
    target = nr_target.reshape(target_shape)
    target = np.rollaxis(target, 1, 2)
    # merge players into teams
    s = list(target.shape)
    s[1] = 3
    new_target = np.zeros(s)
    new_target[:, 0] = target[:, 0]
    new_target[:, 1] = target[:, 1:6].sum(1)
    new_target[:, 2] = target[:, 6:].sum(1)
    new_target[new_target > 1] = 1
    return new_target


def make_reference(x, crop_size, ref_type):
    """
        ref_type: {'bmf', 'bt'}
            bmf: Ball-Mid-Frame, normalize by where the ball is at mid-frame
            tb:  Track-Ball, normalize by where the ball is at each frame
    """

    assert(crop_size[0] % 2 == 1 and crop_size[1] % 2 == 1)
    # x.shape = (N, 11, T, 2)
    if ref_type == 'bmf':
        ball_mid_frame = x[:, 0, x.shape[2] // 2]  # shape = (N,2)
        # shape = (11, T, N, 2)
        ball_mid_frame = np.tile(ball_mid_frame, (11, x.shape[2], 1, 1))
        ball_mid_frame = np.rollaxis(
            ball_mid_frame, 0, 3)  # shape = (T, N, 11, 2)
        ball_mid_frame = np.rollaxis(
            ball_mid_frame, 0, 3)  # shape = (N, 11, T, 2)
    elif ref_type == 'tb':
        ball_mid_frame = x[:, 0]  # shape = (N,T,2)
        ball_mid_frame = np.tile(ball_mid_frame, (11, 1, 1, 1))
        ball_mid_frame = np.rollaxis(ball_mid_frame, 0, 2)
    else:
        raise Exception(
            'either unknown reference type, or just dont use "crop" in config')

    reference = ball_mid_frame
    r0 = np.ceil(crop_size[0] / 2).astype('int32') + 1
    r1 = np.ceil(crop_size[1] / 2).astype('int32') + 1
    reference = reference - np.tile(np.array([r0, r1]),
                                    (x.shape[0], x.shape[1], x.shape[2], 1))
    return reference
