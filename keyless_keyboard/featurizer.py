from keyless_keyboard.config import KeylessConfig
keyless_config = KeylessConfig()
mean = keyless_config.mean
stdev = keyless_config.stdev

# x is an array of 3d points
def featurize(x):
    norm_x = [[(p[0]-mean[0])/stdev[0], (p[1]-mean[1])/stdev[1], (p[2]-mean[2])/stdev[2]] for p in x]
    feat_x = flatten(norm_x)
    return feat_x

# x is a 2d array
def flatten(x):
    return sum(x, [])