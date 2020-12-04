def myweight(distances):
    sigma2 = .4 # we can change this number
    return np.exp(-distances**2/sigma2)