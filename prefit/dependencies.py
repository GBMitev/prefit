# %%
import pandas as pd
import numpy as np
from scipy.stats import chi2
from lmfit import Minimizer, Parameters
from pandarallel    import pandarallel

def ChiSquared(Exp, Obs, ddof):

    deg = len(Exp) - 1 - ddof
    
    chi = sum(((Exp-Obs)**2)/Exp)
    reduced_chi = chi/deg

    pvalue = 1-chi2.cdf(chi, deg)

    Chi2 = {"statistic": reduced_chi, "pvalue":pvalue}
    return Chi2

