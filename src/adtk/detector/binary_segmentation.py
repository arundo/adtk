import numpy as np

def gen_pp(changepoints, intensities):
    """
    Function to simulate Poisson process data (used for testing the segmentation functions)
    
    Parameters
    ----------
    changepoints: list, required
        List of changepoints you want present in the data 
        (including the last item in the list as the duration of the process)
        
    intensities: list, required
        Corresponding intensities of the Poisson process, same length as changepoints
        Note: np.random.exponential uses the inverse parametrization for the time between events,
        so make sure to adjust the intensities accordingly
    """
    
    pp = []
    pp.append(0)
    
    c = 0
    while pp[-1] < changepoints[c]:
        
        t_delta = pp[-1] + np.random.exponential(1 / intensities[c])
        
        if t_delta < changepoints[-1]:
            pp.append(t_delta)
            
            if t_delta >= changepoints[c]:
                c += 1
                
        else: break
            
    return(pp)


def BIC_approx(pp, T):
    """
    Function to calculate the BIC approximation to the posterior odds ratio
    
    Parameters
    ----------
    pp: vector, required
        Vector of timestamps corresponding to each event in the Poisson process. Leave out 0, if present
        
    T: integer, required
        Length (duration) of the process
    """
    
    n = pp.shape[0]
    
    likelihood = 0
    
    changepoint = 0
    changepoint_index = 0
    
    for j in range(0, n - 1):
        
        j_likelihood = (((j + 1) / pp[j])**(j + 1)) * (((n - (j + 1)) / (T - pp[j]))**(n - (j + 1)))
        
        if j_likelihood > likelihood:
            likelihood = j_likelihood
            changepoint = pp[j]
            changepoint_index = j + 1
    
    bic10 = (changepoint_index * np.log(changepoint_index / changepoint)) + \
    ((n - changepoint_index) * np.log((n - changepoint_index) / (T - changepoint))) - \
    (n * np.log(n / T)) - \
    (np.log(n))
    
    return([bic10, changepoint])


def binary_segmentation(pp, T, shift = 0):
    """
    Function to segment the poisson process using the BIC approximation and 
    non-informative prior on the changepoint locations
    
    Returns changepoints and intensities
    
    Parameters
    ----------
    pp: vector, required
        Vector of timestamps corresponding to each event in the Poisson process. Leave out 0, if present
        
    T: integer, required
        Length (duration) of the process
    
    shift: integer, optional
        For recursive feed
    """
    
    n = pp.shape[0]
    
    changepoints = []
    intensities = []
    
    test_stat = BIC_approx(pp, T)
    
    if test_stat[0] > 0:
        
        changepoints.append(test_stat[1] + shift)
        
        pp1 = pp[pp <= test_stat[1]]
        pp2 = pp[pp > test_stat[1]]
        
        s1 = binary_segmentation(pp1, test_stat[1])
        s2 = binary_segmentation(pp2 - test_stat[1], T - test_stat[1], shift + test_stat[1])
        
        changepoints.extend(s1[0] + s2[0])
        intensities.extend(s1[1] + s2[1])
        
    else:
        intensities.append((n + 0.01) / (T + 0.01))
    
    return([changepoints, intensities])
