# %%
from .dependencies import *
from .fit_emo import *
from .fit_mlr import *
# %%
def get_shape_set_permutations(PLs, PRs, NLs, NRs, SameP = True):
    shape_set_permuations = []
    for PL in PLs:
        for PR in PRs:
            for NL in NLs:
                for NR in NRs:
                    shape_set_permuations.append((PL,PR,NL,NR))
    
    df = pd.DataFrame(shape_set_permuations, columns=["PL", "PR","NL","NR"])
    df = df[df["NL"]<=df["NR"]]
    if SameP == True:
        df = df[df["PL"] == df["PR"]]
    
    return df.reset_index(drop = True)

def Optimize_EMO(R,V,cores,PLs,PRs,NLs,NRs,**kwargs):
    
    SameP       = kwargs.get("SameP"        ,True)
    Vary        = kwargs.get("Vary"         ,None)
    Bounds      = kwargs.get("Bounds"       ,None)
    method      = kwargs.get("method"       ,"least_squares")
    full        = kwargs.get("full"         ,False)
    statistics  = kwargs.get("statistics"   ,True)


    df = get_shape_set_permutations(
        PLs,
        PRs,
        NLs,
        NRs,
        SameP=SameP)
    
    pandarallel.initialize(progress_bar=True,nb_workers = cores,verbose = 0)
    names = ["TE",
             "RE",
             "AE",
             "PL",
             "PR",
             "NL",
             "NR",
             *["A"+f"{i}" for i in range(max(NRs)+1)],
             "chi2_statistic",
             "pvalue"] 
    
    fits = pd.DataFrame(columns=[*names])
    df["EMO_Params"] = df.apply(lambda x:EMO_Parameters(
        R,
        V,
        PL = x["PL"],
        PR = x["PR"],
        NL = x["NL"],
        Order = x["NR"], 
        Vary = Vary, 
        Bounds = Bounds
        ), axis = 1)
    
    df[["Fitted","chi2","pvalue"]]     = df.parallel_apply(lambda x: x["EMO_Params"].fit_parameters(full=full,statistics = statistics,method = method), axis = 1, result_type = "expand")

    for fit in df[["Fitted"]].itertuples(index= False):
        fit = fit[0].transpose().reset_index(drop = True)
        fit.columns = fit.iloc[0].to_list()
        fit = fit.iloc[1:]
        fits = pd.concat([fits,fit])

    
    return fits.reset_index(drop = True) if full == False else df
# %%
def Optimize_MLR(r,v,cores, Order_B_list, Order_C_list, P_list, **kwargs):
    pandarallel.initialize(progress_bar=True,nb_workers = cores,verbose = 0)

    Vary        = kwargs.get("Vary"         ,{})
    Bounds      = kwargs.get("Bounds"       ,{})
    method      = kwargs.get("method"       ,"least_squares")
    statistics  = kwargs.get("statistics"   ,True)

    TE = kwargs.get("TE",None)
    RE = kwargs.get("RE",None)
    AE = kwargs.get("AE",None)

    Coulomb = kwargs.get("Coulomb",True)

    shape_set_permutations = []
    for b in Order_B_list:
        for c in Order_C_list:
            for p in P_list:
                shape_set_permutations.append((b,c,p))
    df = pd.DataFrame(shape_set_permutations, columns=["Order_B", "Order_C","P"])

    names = ["TE",
             "RE",
             "AE",
             "P" ,
             *[f"B{i}" for i in range(max(Order_B_list)+1)],
             *[f"C{i}" for i in range(max(Order_C_list)+1)],
             "chi2_statistic",
             "pvalue"] 
    
    fits = pd.DataFrame(columns=[*names])

    df["MLR_Params"] = df.apply(lambda x:MLR(r,v,
                                             TE=TE,
                                             RE=RE,
                                             AE=AE,
                                             P=x["P"],
                                             Order_B=x["Order_B"],
                                             Order_C=x["Order_C"],
                                             Coulomb=Coulomb,
                                             Vary = Vary,
                                             Bounds = Bounds),axis=1)
    
    df[["Fitted","chi2","pvalue"]] = df.parallel_apply(lambda x: x["MLR_Params"].fit_parameters(full=False,statistics = statistics,method = method), axis = 1, result_type = "expand")
    
    for fit in df[["Fitted"]].itertuples(index= False):
        fit = fit[0].transpose().reset_index(drop = True)
        fit.columns = fit.iloc[0].to_list()
        fit = fit.iloc[1:]
        fits = pd.concat([fits,fit])
    
    return fits.sort_values("chi2_statistic").reset_index(drop = True)
