# %%
from .dependencies import *
from .poten import mlr
from .utils import *
class MLR:
    def __init__(self, TE, RE, AE, P, BINF, Order_B, Order_C, Coulomb = True):
        # Attributes
        self.TE     = TE
        self.RE     = RE
        self.AE     = AE
        self.P      = P
        self.Order_B = Order_B
        self.Order_C = Order_C
        self.Coulomb = Coulomb

        mlr_params = {
            "TE":[TE],
            "RE":[RE],
            "AE":[AE],
            "P" :[int(P)]
            }

        
        beta_expansion = np.append(np.array(1), np.linspace(0,0,Order_B))
        for i in range(Order_B+1):
            mlr_params[f"B{i}"] = [beta_expansion[i]]

        Coulomb_C1 =  hartree_to_cm(1)*bohr_to_angstrom(1)
        LR_expansion = np.append(np.array([0,Coulomb_C1]), np.linspace(0,0,Order_C-1)) \
            if Coulomb == True else \
            np.linspace(0,0,Order_C+1)
        
        for i in range(Order_C+1):
            mlr_params[f"C{i}"] = [LR_expansion[i]]

        self.beta_expansion = beta_expansion
        self.LR_expansion   = LR_expansion

        mlr_params = pd.DataFrame(mlr_params).transpose().reset_index()
        mlr_params.columns = ["Param","Value"]

        self.params = mlr_params

class MLR_Parameters(MLR):
    def __init__(self,R, V, TE=None, RE=None, AE=None, P = 1, BINF = 1,Order_B=1, Order_C = 1, Coulomb = True,Vary = None, Bounds = None):
        from .poten import mlr

        self.mlr = mlr

        RE = R[[*V].index(min(V))] if RE is None else RE
        TE = min(V) if TE is None else TE
        AE = V[-1] if AE is None else AE

        self.R = R
        self.V = V

        super().__init__(TE, RE, AE, P, BINF, Order_B, Order_C, Coulomb=Coulomb)

        def Fill_Out(parameter, Type):
            fill = {"vary":[False, True],
                    "bounds":[[-np.inf,np.inf],[-np.inf,np.inf]]}
            if parameter == None or len(parameter) == 0:
                parameter = [fill[Type][0]]*3+\
                [fill[Type][1] for i in range(Order_B+1)]+[fill[Type][0]] \
                
                parameter += [fill[Type][0]] + [fill[Type][1]] + [fill[Type][1] for i in range(Order_C-1)]\
                    if self.Coulomb == True else \
                        [fill[Type][1] for i in range(Order_C+1)]
                
            elif len(parameter) > 0:
                while len(parameter) < 3:
                    parameter.append(fill[Type][0])
                while len(parameter) < 3+Order_B+1:
                    parameter.append(fill[Type][1])
                parameter.append(fill[Type][0])
                while len(parameter) < 5 + Order_B + Order_C:
                    parameter.append(fill[Type][1])

            shape     = [fill[Type][0]]
            parameter = [*parameter[:3],*shape,*parameter[3:]][:Order_B+Order_C+6]
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
        def Residual(param_dict, R, V):
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
        minner = Minimizer(Residual, params, fcn_args = (R,V))
        result = minner.minimize(method=method)

        if full == True:
            return result
        else: 
            param_dict = result.params.valuesdict()
            return param_dict
        
    def fit_parameters(self, sigma=1, full=False, statistics = True, method = 'leastsq'):
        param_dict = self.get_param_dict(fitting = True)
        
        fitted = self.fitter(param_dict,self.R, self.V, self.mlr, sigma = sigma, full = full, method=method)
        if full == False:
            calc   = self.mlr(self.R,fitted)
            ddof   = 4 + self.Order_B + self.Order_C
            chi2   = ChiSquared(self.V, calc, ddof)

            test_statistic = chi2["statistic"]
            pvalue         = chi2["pvalue"]

            fitted["chi2_statistic"] = test_statistic
            fitted["pvalue"]         = pvalue

            fitted = pd.DataFrame(fitted, index = [0]).transpose().reset_index()
            fitted.columns = ["Param", "Value"]

            return fitted, test_statistic, pvalue if statistics == True else fitted