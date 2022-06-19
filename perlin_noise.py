# From https://github.com/pvigier/perlin-numpy
# with some minor modifications
import numpy as np
import cv2


def generate_perlin_noise_2d(shape, res, rng=None):
    # seed = int(time.time()) if seed is None else seed
    rng = np.random.RandomState() if rng is None else rng

    def f(tt):
        return 6 * tt ** 5 - 15 * tt ** 4 + 10 * tt ** 3

    d = (shape[0] // res[0], shape[1] // res[1])
    d_res = (shape[0] % res[0], shape[1] % res[1])
    # grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    grid = np.mgrid[0:shape[0], 0:shape[1]].transpose(1, 2, 0) / shape * res % 1

    # Gradients
    angles = 2 * np.pi * rng.rand(res[0] + 1, res[1] + 1)
    # angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = cv2.resize(gradients[0:-1, 0:-1], dsize=(shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    g10 = cv2.resize(gradients[1:, 0:-1], dsize=(shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    g01 = cv2.resize(gradients[0:-1, 1:], dsize=(shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    g11 = cv2.resize(gradients[1:, 1:], dsize=(shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    # g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    # g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    # g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # if d_res[0]:
    #     g00 = np.concatenate((g00, g00[0:d_res[0], :]), axis=0)
    #     g10 = np.concatenate((g10, g10[0:d_res[0], :]), axis=0)
    #     g01 = np.concatenate((g01, g01[0:d_res[0], :]), axis=0)
    #     g11 = np.concatenate((g11, g11[0:d_res[0], :]), axis=0)
    # if d_res[1]:
    #     g00 = np.concatenate((g00, g00[:, 0:d_res[1]]), axis=1)
    #     g10 = np.concatenate((g10, g10[:, 0:d_res[1]]), axis=1)
    #     g01 = np.concatenate((g01, g01[:, 0:d_res[1]]), axis=1)
    #     g11 = np.concatenate((g11, g11[:, 0:d_res[1]]), axis=1)
    # Ramps

    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5, rng=None):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise *= amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]), rng)
        frequency *= 2
        amplitude *= persistence
    return noise


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(0)
    perlin_noise = generate_perlin_noise_2d((256, 256), (8, 8))
    plt.imshow(perlin_noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()

    np.random.seed(0)
    fractal_noise = generate_fractal_noise_2d((256, 256), (8, 8), 5)
    plt.figure()
    plt.imshow(fractal_noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.show()
