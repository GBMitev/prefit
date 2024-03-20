# %%
from .dependencies import *
from .poten import emo

class EMO:
    def __init__(self, TE, RE, AE, PL, PR, NL, Order, Expansion=None):
        # Attributes
        self.TE = TE
        self.RE = RE
        self.AE = AE
        self.PL = PL
        self.PR = PR
        self.NL = NL
        self.Order = Order
        
        emo_params = {
            "TE":[TE],
            "RE":[RE],
            "AE":[AE],
            "PL":[int(PL)],
            "PR":[int(PR)],
            "NL":[int(NL)],
            "NR":[int(Order)]
        }

        if Expansion is None:
            Expansion_Values = np.append(np.array(1), np.linspace(0,0,Order))
        elif Expansion is not None and len(Expansion) == Order+1:
            Expansion_Values = Expansion
        elif Expansion is not None and len(Expansion) != Order+1:
            raise ValueError("Expansion does not have the same number of coefficients as the Order")
        
        self.Expansion = Expansion_Values

        for i in range(Order+1):
            emo_params["A"+str(i)] = [Expansion_Values[i]]

        emo_params = pd.DataFrame(emo_params).transpose().reset_index()
        emo_params.columns = ["Param","Value"]

        self.params = emo_params

class EMO_Parameters(EMO):
    def __init__(self,R, V, TE=None, RE=None, AE=None, PL=1, PR=1, NL=1, Order=1, Expansion=None, Vary = None, Bounds = None):
        from .poten import emo

        self.emo = emo

        RE = R[[*V].index(min(V))] if RE is None else RE
        TE = min(V) if TE is None else TE
        AE = V[-1] if AE is None else AE

        self.R = R
        self.V = V

        super().__init__(TE, RE, AE, PL, PR, NL, Order, Expansion=Expansion)

        def Fill_Out(parameter, Type):
            fill = {"vary":[False, True],
                    "bounds":[[-np.inf,np.inf],[-np.inf,np.inf]]}

            if parameter == None or len(parameter) == 0:
                parameter = [fill[Type][0]]*3+[fill[Type][1] for i in range(self.Order+1)] 
            elif len(parameter) > 0:
                while len(parameter) < 3:
                    parameter.append(fill[Type][0])
                while len(parameter) < 3+self.Order+1:
                    parameter.append(fill[Type][1])

            shape     = [fill[Type][0]]*4
            parameter = [*parameter[:3],*shape,*parameter[3:]][:self.Order+8]
            return parameter

        Vary = Fill_Out(Vary, Type = "vary")      
        Bounds = Fill_Out(Bounds, Type = "bounds")

        params = self.params[["Param","Value"]]
        params["Vary"] = Vary
        params[["Upper_Bound","Lower_Bound"]] = Bounds
        
        self.fitting_params = params

    def get_param_dict(self, fitting = True):
        param_dict = {}
        if fitting == True:
            for Param, Value, Vary, Lower_Bound, Upper_Bound in self.fitting_params.itertuples(index = False):
                param_dict[Param] = [Value, Vary, Lower_Bound, Upper_Bound]
        else:
            for Param, Value in self.params[["Param","Value"]].itertuples(index = False):
                param_dict[Param] = Value
        
        return param_dict
    
    def fitter(self, param_dict, R, V, f,sigma = 1.0,full = True,method='leastsq'):
        '''
        Function to fit a parameter dictionary to a set of ab initio values

        Inputs:
            ParamDict  = parameter dictionary           , dtype = N/A  , data structure = dict
            R          = ab initio internuclear distance, dtype = float, data structure = np.ndarray
            V          = ab initio potential energy     , dtype = float, data structure = np.ndarray
            f          = function to fit to             , dtype = N/A  , data structure = N/A (function)
            sigma      = fit weights                    , dtype = float, data structure = value or np.ndarray
            full       = print full fit result or values, dtype = bool , data structure = value
        Outputs:
            result     = lmfit fit result               , dtype = N/A, data structure = N/A
            or
            param_dict = lmfit fit result               , dtype = N/A, data structure = dict
        '''
        def Residual(param_dict, R, V,sigma):
            '''
            Wrapper function to measure residuals
            
            Inputs:
                ParamDict  = parameter dictionary           , dtype = N/A  , data structure = dict
                R          = ab initio internuclear distance, dtype = float, data structure = np.ndarray
                V          = ab initio potential energy     , dtype = float, data structure = np.ndarray

            Outputs:
                Resid      = residuals of parameterisation  , dtype = float, data structure = np.ndarray
            '''
            Resid = ((f(R, param_dict.valuesdict())-V))/sigma
            return Resid
        
        # initialise params as lmfit Parameters() object
        params = Parameters()

        # add all values to Parameters object from ParamDict
        for key,value in param_dict.items():
            params.add(key, value = value[0], vary=value[1],min = value[2], max = value[3])

        # initialise Minimizing routine
        minner = Minimizer(Residual, params, fcn_args = (R,V,sigma))
        result = minner.minimize(method=method)

        if full == True:
            return result
        else: 
            param_dict = result.params.valuesdict()
            return param_dict
        
    def fit_parameters(self, sigma=1, full=False, statistics = True, method = 'leastsq'):
        param_dict = self.get_param_dict(fitting = True)
        
        # try:
        fitted = self.fitter(param_dict,self.R, self.V, self.emo, sigma = sigma, full = full, method=method)
        if full == False:
            calc   = self.emo(self.R,fitted)
            ddof   = 8 + self.Order
            chi2   = ChiSquared(self.V, calc, ddof)

            test_statistic = chi2["statistic"]
            pvalue         = chi2["pvalue"]

            fitted["chi2_statistic"] = test_statistic
            fitted["pvalue"]         = pvalue

            fitted = pd.DataFrame(fitted, index = [0]).transpose().reset_index()
            fitted.columns = ["Param", "Value"]

            return fitted, test_statistic, pvalue if statistics == True else fitted
        else:
            return fitted    
        # except:
        #     fitted = np.nan, -1, -1
        #     return fitted
            
        
            