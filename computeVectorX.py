import numpy as np

def computeVectorX(fixed_a, fixed_b, alpha, beta, ell, k):
    gamma = 0.5
    x_dist = np.zeros(k)
    power = 0.5
    t_max = 10

    for _ in range(t_max):
        f_a = fixed_a
        f_b = fixed_b
        power /= 2

        temp = (1 - gamma) * beta * ell / (gamma * alpha + (1 - gamma) * beta)
        x_dist = temp
        f_a += np.sum(alpha * temp * temp)
        temp = ell - temp
        f_b += np.sum(beta * temp * temp)

        if f_a == f_b:
            break

        gamma = gamma + power if f_a > f_b else gamma - power

    return x_dist
