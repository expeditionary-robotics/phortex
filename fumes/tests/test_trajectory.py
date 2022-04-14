from fumes.trajectory import Spiral, Lawnmower

def test_lawnmower_init():
    path = Lawnmower(0, 0.5, 10, 10, 2)
    samples = path.uniformly_sample(0.5)
    samples2 = path.path_sample(0.5)

def test_lawnmower_noisy():
    path = Lawnmower(0, 0.5, 10, 10, 2, noise=0.1)
    samples = path.uniformly_sample(0.5)
    samples2 = path.path_sample(0.5)

def test_sprial_init():
    path = Spiral(0, 0.5, 10, 10, 2)
    samples = path.uniformly_sample(0.5)
    samples2 = path.path_sample(0.5)

def test_sprial_noisy():
    path = Spiral(0, 0.5, 10, 10, 2, noise=0.1)
    samples = path.uniformly_sample(0.5)
    samples2 = path.path_sample(0.5)
