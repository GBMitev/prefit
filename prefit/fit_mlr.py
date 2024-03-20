# %%
from .dependencies import *
from .utils import *
from .poten import mlr

class MLR:
    def __init__(self, r,v, TE=None, RE=None, AE=None, P=1, Order_B=0, Order_C=1, Coulomb = True, **kwargs):
        self.r = r
        self.v = v

        self.TE     = min(v) if TE is None else TE
        self.RE     = r[[*v].index(min(v))] if RE is None else RE
        self.AE     = v[-1] if AE is None else AE
        self.P      = P

        self.Order_B = Order_B
        self.Order_C = Order_C
        self.Coulomb = Coulomb
        
        mlr_params = {
            "TE":[self.TE],
            "RE":[self.RE],
            "AE":[self.AE],
            "P" :[int(self.P)]
            }
        
        B = {f"B{i}":kwargs.get(f"B{i}", [1] if i == 0 else [0]) for i in np.arange(0,Order_B+1,1)}
        if Coulomb==True:
            Coulomb_Constant = [hartree_to_cm(1)*bohr_to_angstrom(1)]
            C = {f"C{i}":Coulomb_Constant if i == 1 else [0] for i in np.arange(0,Order_C+1,1)}
        else:
            C = {f"C{i}":kwargs.get(f"C{i}", [0]) for i in np.arange(0,Order_C+1,1)}

        mlr_params.update(B)
        mlr_params.update(C)

        self.params = mlr_params

        # Fitting parameters
        Vary = kwargs.get("Vary",{})
        Bounds = kwargs.get("Bounds",{})

        shape_set = ["TE","RE","AE","P"]

        for key in self.params.keys():

            #Getting vary values 
            if key in shape_set:
                self.params[key].append(Vary.get(key, False))
            else:
                self.params[key].append(Vary.get(key, True))

            #Getting bound values
            self.params[key] = self.params[key] + Bounds.get(key, [-np.inf, np.inf])
        
        parameters = pd.DataFrame.from_dict(self.params, orient = "index", columns = ["Value","Vary","LBound","UBound"])
        parameters["Parameter"] = parameters.index
        self.parameters = parameters[["Parameter", *parameters.columns[:-1]]].reset_index(drop = True)

    def fitter(self, param_dict, r, v, f,sigma = 1.0,full = True,method='leastsq'):
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
        def Residual(param_dict, r, v):
            '''
            Wrapper function to measure residuals
            
            Inputs:
                ParamDict  = parameter dictionary           , dtype = N/A  , data structure = dict
                R          = ab initio internuclear distance, dtype = float, data structure = np.ndarray
                V          = ab initio potential energy     , dtype = float, data structure = np.ndarray

            Outputs:
                Resid      = residuals of parameterisation  , dtype = float, data structure = np.ndarray
            '''
            Resid = ((f(r, param_dict.valuesdict())-v))/sigma
            return Resid
        
        # initialise params as lmfit Parameters() object
        params = Parameters()

        # add all values to Parameters object from ParamDict
        for key,value in param_dict.items():
            params.add(key, value = value[0], vary=value[1],min = value[2], max = value[3])

        # initialise Minimizing routine
        minner = Minimizer(Residual, params, fcn_args = (r,v))
        result = minner.minimize(method=method)

        if full == True:
            return result
        else: 
            param_dict = result.params.valuesdict()
            return param_dict
    
    def fit_parameters(self, sigma=1, full=False, statistics = True, method = 'leastsq'):
        param_dict = self.params
        
        fitted = self.fitter(param_dict, self.r, self.v, mlr, sigma = sigma, full = full, method=method)
        if full == False:
            calc   = mlr(self.r,fitted)
            ddof   = 4 + self.Order_B + self.Order_C
            chi2   = ChiSquared(self.v, calc, ddof)

            test_statistic = chi2["statistic"]
            pvalue         = chi2["pvalue"]

            fitted["chi2_statistic"] = test_statistic
            fitted["pvalue"]         = pvalue

            fitted = pd.DataFrame(fitted, index = [0]).transpose().reset_index()
            fitted.columns = ["Param", "Value"]

            return fitted, test_statistic, pvalue if statistics == True else fitted