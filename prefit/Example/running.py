# %%
R, V = np.loadtxt("/home/gmitev/Documents/Papers/SupplementaryData/05LoGr/CSVs/VA2S.csv", delimiter = ",",unpack = True, skiprows = 1)

Vary = [
    False,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True
    ]

Bounds = [
    [-np.inf,np.inf],
    [1.8,2],
    [-np.inf,np.inf],
    [-10,10],
    [-1,1],
    [-1,1],
    [-1,1],
    [-1,1],
    [-1,1],
    [-1,1],
    [-1,1],
    [-1,1]
    ]

PLs = [2,4]
PRs = [2,4]
NLs = [1,2,3]
NRs = [3,4,5,6]
fits = Optimize_EMO(
    R,
    V,
    16,
    PLs, 
    PRs, 
    NLs, 
    NRs, 
    Vary = Vary, 
    Bounds = Bounds,
    full = False, 
    method="least_squares",
    SameP=False)
# %%
fits.sort_values("chi2_statistic")