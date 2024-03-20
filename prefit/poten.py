# %%
from .dependencies import *

def emo(r, params):
    '''
    Extended Morse Oscillator function taking inputs of internuclear distance (r) and a parameter dictionary (see inputs).

    Inputs:
        r          = internuclear distance, dtype = float, data structure = np.ndarray
        params     = parameter dictionary, must contain the following keys and values:

            "RE"   = equillibrium bond length, dtype = float, data structure = value
            "TE"   = potential minimum,        dtype = float, data structure = value
            "AE"   = asymptotic energy,        dtype = float, data structure = value
            "NL"   = left side sum limit,      dtype = int  , data structure = value
            "PL"   = left side Surkus param    dtype = int  , data structure = value
            "PR"   = right side Surkus param   dtype = int  , data structure = value
            "A[N]" = Morse parameter (see *)   dtype = float, data structure = value

            * Morse parameter coefficients are of format "A[N]" where [N] represents a natural number. 
            All [N] beneath the maximum [N] must exist e.g. [A0, A1, A2, A3], not [A0, A2, A3]

    Outputs:
        V          = EMO Potential Energy,     dtype = float, data structure = np.ndarray
    '''
    
    # Checking that r is an array
    if type(r) != np.ndarray:
        return "R needs to be of type:array"
    else:
        # Unpacking parameters
        RE, TE, AE, NL, PL, PR  = params["RE"], params["TE"], params["AE"], params["NL"], params["PL"], params["PR"]
        NL, PL, PR = int(NL), int(PL), int(PR)

        A = np.array([params.get(f"A{i}",0) for i in np.arange(0,10,1)])

        # Finding index where R_left changes to R_right
        R_step = [*filter(lambda x: x>= RE, list(r))][0]
        index_limit = list(r).index(R_step)

        # Calculating Surkus variable
        Surkus_conditions   = [r<= RE, r>RE]
        Surkus_values       = [(r**PL-RE**PL)/(r**PL+RE**PL),(r**PR-RE**PR)/(r**PR+RE**PR)]
        
        S = np.select(Surkus_conditions, Surkus_values)
        S_list = np.tile(S,(np.size(A),1))
        S_list = [sval**idx for idx, sval in enumerate(S_list)]

        # Forming Morse parameter (combining Surkus variable and expansion coefficients)
        Beta = S_list*A.reshape((-1,1))
        Beta = [np.where(idx >= index_limit,np.sum(BetaVal), np.sum(BetaVal[:NL+1])) for idx, BetaVal in enumerate(Beta.T)]

        #Calculating potential
        exponent = -1*(np.array(Beta))*(r-RE)
        V = TE + (AE - TE)*(1-np.exp(exponent))**2
        return V

def mlr(r, params):
    # getting parameters
    TE, AE, RE, P = params["TE"],params["AE"],params["RE"],params["P"]

    B = np.array([params.get(f"B{i}",0) for i in np.arange(0,10,1)])
    C = np.array([params.get(f"C{i}",0) for i in np.arange(0,10,1)])

    # calculating De
    DE = AE-TE

    # Long range
    RE_list = np.tile(1/RE,(np.size(C),1))
    RE_list = [rval**idx for idx, rval in enumerate(RE_list)]

    r_list  = np.tile(1/r ,(np.size(C),1))
    r_list  = [rval**idx for idx, rval in enumerate(r_list)]

    U_RE = np.array([
        np.sum(val) for val in (RE_list*C.reshape((-1,1))).T
            ])[0]
    
    U_r = np.array([
        np.sum(val) for val in (r_list*C.reshape((-1,1))).T
            ])

    if U_RE != 0:
        U_LR = U_r/U_RE
    else:
        raise ValueError("At least one long range parameter (C(i)) must be none-zero")
    
    # beta expansion
    Surkus = (r**P-RE**P)/(r**P+RE**P)
    S_list = np.tile(Surkus,(np.size(B),1))
    S_list = [sval**idx for idx, sval in enumerate(S_list)]

    beta_expansion = S_list*B.reshape((-1,1))
    beta_expansion = np.array([np.sum(BetaVal) for BetaVal in beta_expansion.T])

    BINF = np.log(2*DE/(U_RE))

    beta = Surkus*BINF + (1-Surkus)*beta_expansion
    beta *= -1
    
    # combining
    unscaled = (1 - U_LR*np.exp(beta*Surkus))**2
    V        = TE + DE*unscaled

    return V
# %%
