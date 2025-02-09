#!/usr/bin/python
"""
Module containing tree growth information and allometric functions

class Growth:   for tree growth data and fitting
def ryan      allometric function based on C. Ryan biotropica paper (2010)

"""

import sys
import os
import logging as log
import pdb
import csv

import numpy as np
import math
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

from . import configuration, io_

from .tree_params import TreeParams

class TreeGrowth(object):

    """
    Object holding tree growth data, and fit functions/params for this.
    Also includes method for printing and plotting data and fits

    Instance variables
    ----------------
    tree_params TreeParams object (holds carbon and dens, among others)
    func        dict holding the four fitting functions
    funcDeriv   dict holding the four derivative functions
    age         tree age data in years
    diam        tree diameter data in cm
    best        string with best fit ('exp, 'log', 'lin, or 'hyp')
    fitDiam     dict holding fit diameter data in cm for all four fits 
    fitParams   dict holding fitting params for all four fits
    fitMSE      dict holding MSE for all four fits

    """
    
    def __init__(self, tree_params, growth_params, allom='chave dry'):
        """Initialise tree growth data.
        
        Args: 
            growth_params: dict with growth params 
                           keys=are 'age','diam' OR 'age','biomass'
            tree_params: tree_params.TreeParams object (holds tree params)
            allom: which allometric to use - ignored if biomass is a key
                   in growth_params since it won't be used
        Raises:
            KeyError: if dict doesn't have the right keys
            KeyError: if allom isn't in the allometric dict
        
        """
        
        try:
            self.age = growth_params['age']
            self.diam = growth_params['diam']
            allom = allom.lower()
            self.allom = allom
        except KeyError:
            try:
                self.age = growth_params['age']
                self.biomass = growth_params['biomass']
            except KeyError:
                log.exception("Growth parameters not specified properly")
                sys.exit(1)

        if 'biomass' not in growth_params:
            self.biomass = np.zeros(len(self.diam))
            try:
                for i in range(len(self.diam)):
                    self.biomass[i] = allometric[self.allom](
                            self.diam[i], tree_params)
            except KeyError:
                log.exception("Allometric is not recognized")
                sys.exit(1)

        # Dictionaries for fitting functions and derivatives
        self.func = {
                'exp': self.exp_fn, 'hyp': self.hyp_fn,
                'lin': self.lin_fn, 'log': self.log_fn
        }
        self.funcDeriv = {
                'exp': self.exp_fn_deriv, 'hyp': self.hyp_fn_deriv,
                'lin': self.lin_fn_deriv, 'log': self.log_fn_deriv
        } 

        # Find fits
        self.allFitData, self.allFitParams, self.allMse = self.fit()
        
        # Find key with best fit
        self.best = min(self.allMse, key=self.allMse.get)
        self.fitData = self.allFitData[self.best]
        self.fitParams = self.allFitParams[self.best]
        self.fitMse = self.allMse[self.best]
    
        
    
    @classmethod
    def from_csv1(
            cls, tree_params, n, filename='growth_measurements.csv', 
            isBiomass=False, allom='chave dry'):
        """Construct Growth object using data in a csv file.
        
        When using intpu_csv from command line, constructing dictionary of 
        np.arrays for each new cohort
        
        Args:
            tree: Tree object with tree information
            filename: csv to read data from with columns age,diam
            True if data in column 2 is biomass (not dbh)
            allom: which allometric to use
        Returns:
            Growth object
        Raises:
            IndexError: if file isn't the right format

        """  

        data = pd.read_csv(configuration.INP_DIR + "/" + filename, 
                           sep = ',')
        reader = data.loc[n]
        dictionary = reader.to_dict()
        
        age_input = ['age1','age2','age3','age4','age5','age6']
        age = {key: dictionary[key] for key in age_input}
        age = np.array(list(age.values())).astype(float) #https://stackoverflow.com/questions/45957968/float-arguments-and-dict-values-with-numpy
        age = np.array(sorted(age, key=int))
                
        diam_input = ['diam1','diam2','diam3','diam4','diam5','diam6']
        diam = {key: dictionary[key] for key in diam_input}
        diam = np.array(list(diam.values())).astype(float)
                
        params = {
        'age': age,
        'diam': diam,             
        }        


        try:
            params = {
                        'age': age,
                        'diam': diam,             
                }


            growth = cls(tree_params, params, allom)

        except IndexError:
            log.exception("Can't read growth data from %s " % filename)
            sys.exit(1)

        return growth
                

    @classmethod
    def from_csv2(
            cls, tree_params, n, filename='growth_measurements.csv', 
            isBiomass=False, allom='chave dry'):
        """Construct Growth object using data in a csv file.
        
        When using intpu_csv from command line, constructing dictionary of 
        np.arrays for each new cohort        
        
        Args:
            tree: Tree object with tree information
            filename: csv to read data from with columns age,diam
            True if data in column 2 is biomass (not dbh)
            allom: which allometric to use
        Returns:
            Growth object
        Raises:
            IndexError: if file isn't the right format

        """          
        
        data = pd.read_csv(configuration.INP_DIR + "/" + filename, 
                           sep = ',')
        reader = data.loc[n]
        dictionary = reader.to_dict()
        
        age_input = ['sp2_age1','sp2_age2','sp2_age3','sp2_age4','sp2_age5','sp2_age6']
        age = {key: dictionary[key] for key in age_input}
        age = np.array(list(age.values())).astype(float) 
        age = np.array(sorted(age, key=int))
                
        diam_input = ['sp2_diam1','sp2_diam2','sp2_diam3','sp2_diam4','sp2_diam5','sp2_diam6']
        diam = {key: dictionary[key] for key in diam_input}
        diam = np.array(list(diam.values())).astype(float)
                
        params = {
        'age': age,
        'diam': diam,             
        } 


        try:
            params = {
                        'age': age,
                        'diam': diam,             
                }


            growth = cls(tree_params, params, allom)

        except IndexError:
            log.exception("Can't read growth data from %s " % filename)
            sys.exit(1)

        return growth

    @classmethod
    def from_csv3(
            cls, tree_params, n, filename='growth_measurements.csv', 
            isBiomass=False, allom='chave dry'):
        """Construct Growth object using data in a csv file.
        
        When using intpu_csv from command line, constructing dictionary of 
        np.arrays for each new cohort        
        
        Args:
            tree: Tree object with tree information
            filename: csv to read data from with columns age,diam
            True if data in column 2 is biomass (not dbh)
            allom: which allometric to use
        Returns:
            Growth object
        Raises:
            IndexError: if file isn't the right format

        """          
        
        data = pd.read_csv(configuration.INP_DIR + "/" + filename, 
                           sep = ',')
        reader = data.loc[n]
        dictionary = reader.to_dict()
        
        age_input = ['sp3_age1','sp3_age2','sp3_age3','sp3_age4','sp3_age5','sp3_age6']
        age = {key: dictionary[key] for key in age_input}
        age = np.array(list(age.values())).astype(float) 
        age = np.array(sorted(age, key=int))
                
        diam_input = ['sp3_diam1','sp3_diam2','sp3_diam3','sp3_diam4','sp3_diam5','sp3_diam6']
        diam = {key: dictionary[key] for key in diam_input}
        diam = np.array(list(diam.values())).astype(float)
                
        params = {
        'age': age,
        'diam': diam,             
        } 


        try:
            params = {
                        'age': age,
                        'diam': diam,             
                }


            growth = cls(tree_params, params, allom)

        except IndexError:
            log.exception("Can't read growth data from %s " % filename)
            sys.exit(1)

        return growth        
   
    @classmethod
    def from_arrays(
            cls, tree, age, data, isBiomass=False, allom='chave dry'):
        """Construct Growth object using data from numpy arrays."""
        pass

    def fit(self):
        """Fit tree growth data using four fitting functions. 
        
        Returns:
            **all dicts with keys 'lin','log','hyp','exp'**
            data: data points from each of the fits
            params: fitting parameters
            mse: mean-square error
        Raises:
            RuntimeError: if curve_fit can't fit the data to a curve     
        
        """

        data = {}
        params = {}
        mse = {}
        init = {'exp': [20], 'hyp': [200,0.1],
                'lin': [1], 'log': [100, 0,0]}
        numParams = {'exp': 1, 'hyp': 2, 
                     'lin': 1, 'log': 3}

        for curve in self.func:
            # Do the fitting
            try:
                fitParams = optimize.curve_fit(self.func[curve], self.age,
                                               self.biomass, init[curve]
                )
                par = fitParams[0]
                params[curve] = par
                
            except RuntimeError:
                log.warning("Could not fit data to %s",curve)
                fitParams = None
                par = None

            # Find data corresponding to those fitting params
            # Set data and params to array of nan if there's no fit
            if curve == 'log':
                if fitParams is not None:
                    data[curve] = self.func[curve](
                            self.age, par[0], par[1], par[2]
                    )
                else:
                    params[curve] = np.array(3 * [np.nan])
                    data[curve] = np.array(len(self.age) * [np.nan])

            elif curve == 'lin' or curve == 'exp':
                if fitParams is not None:
                    data[curve] = self.func[curve](self.age, par[0])
                else:
                    params[curve] = np.array(1 * [np.nan])
                    data[curve] = np.array(len(self.age) * [np.nan])
            else:   # hyp
                if fitParams is not None:
                    data[curve] = self.func[curve](
                            self.age, par[0], par[1])
                else:
                    params[curve] = np.array(2 * [np.nan])
                    data[curve] = np.array(len(self.age) * [np.nan])
        
            # Find mse
            # set to inf if there's no fit for a given curve
            if fitParams is not None:
                mse[curve] = self.mse_fn(self.biomass, data[curve])
            else:
                mse[curve] = np.inf

        return data,params,mse
    

    # Functions to fit to
    def hyp_fn(self, x, a, b):
        return a * (1 - np.exp(-b*x))
    
    def hyp_fn_inv(self, y):
        if math.fabs(y) < 0.00000001:
            x = 0
        else:
            a = self.fitParams[0]
            b = self.fitParmas[1]
            if y > a:
                x = a
            else:
                x = (math.log(a) - math.log(a-y)) / b

        return x 

    def hyp_fn_deriv(self, x):
        a = self.fitParams[0]
        b = self.fitParams[1]
        return a * b * np.exp(-b*x)
    
    def lin_fn(self, x, a):
        return a * x 
    
    def lin_fn_inv(self, y):
        if math.fabs(y) < 0.00000001:
            x = 0
        else:
            a = float(self.fitParams[0])     # just to be safe
            x = y / a

        return x

    def lin_fn_deriv(self, y):
        x = self.lin_fn_inv(y)

        a = self.fitParams[0]
        return a

    def log_fn(self, x, a, b, c):
        return a / (1 + np.exp(-b*(x-c)))
   
    def log_fn_inv(self, y):
        if math.fabs(y) < 0.00000001:
            x = 0
        else:
            a = self.fitParams[0]
            b = self.fitParams[1]
            c = self.fitParams[2]
            if y > a:   # should tend towards a
                x = a
            else:
                x = c + (math.log(y)-math.log(a-y)) / b

        return x

    def log_fn_deriv(self, y): 
        x = self.log_fn_inv(y)
        
        a = self.fitParams[0]
        b = self.fitParams[1]
        c = self.fitParams[2]
        ret = (a * b * np.exp(-b*(x-c))) / ((np.exp(-b*(x-c)) + 1)**2)
        
        return ret

    def exp_fn(self, x, a):
        return (1+a)**x -1
    
    def exp_fn_inv(self, y):
        if math.fabs(y) < 0.00000001:
            x = 0
        else:
            a = self.fitParams[0]
            x = math.log(y+1) / math.log(a+1)

        return x

    def exp_fn_deriv(self, x):
        a = self.fitParams[0]
        return ((1+a)**x) * (np.log(1+a))

    def mse_fn(self, yMean, yReal):
        return ((yMean - yReal)**2).mean()

    def plot_(self, fit=True, saveName=None):
        """Plot growth data and all four fits in a matplotlib figure."""
        
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Tree Growth')

        ax = fig.add_subplot(1,1,1)

        ax.plot(self.age, self.biomass, 'o', label='data')

        if fit:
            for curve in self.allFitData:
                ax.plot(self.age, self.allFitData[curve],
                        '-', label="%s fit" % curve
                )

            ax.legend(loc='best')

        ax.set_title('Tree biomass vs. Age')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Biomass (kg C /ha) ')

        # Shift ticks so points not cut off
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xmin = xticks[0] - 0.5*(xticks[1]-xticks[0])
        xmax = xticks[-1] + 0.5*(xticks[-1]-xticks[-2])
        ymin = yticks[0] - 0.5*(yticks[1]-yticks[0])
        ymax = yticks[-1] + 0.5*(yticks[-1]-yticks[-2])
        ax.set_xlim(xmin, xmax)

        if saveName is not None:
            plt.savefig(os.path.join(configuration.OUT_DIR, saveName))

    def print_(self, fit=None, params=None, mse=None):
        """Print data and fits for tree growth data to stdout."""
        fit = True
        params = True
        mse = True

        
        print ("\n\nTREE GROWTH DATA")
        print ("=========\n")
        print("Allometric: ",self.allom)
        print ("\n  Age    Diameter  Biomass")
        print ("(years)    (cm)     (kg C)")
        print ("--------------------------")
        for i in range(len(self.age)):
            print(("  %2d      %5.2f     %6.2f" % (
                    self.age[i], self.diam[i], self.biomass[i])))
        if fit:
            print ("\n Data      Exp.     Hyp.     Lin.     Log.")
            print ("-----------------------------------------")
            for i in range(len(self.age)):
                print(("%6.2f   %6.2f   %6.2f   %6.2f   %6.2f" % (
                        self.biomass[i], 
                        self.allFitData['exp'][i],
                        self.allFitData['hyp'][i],
                        self.allFitData['lin'][i], 
                        self.allFitData['log'][i]
                )))
        if params and mse:
            print(("\nMSE      %6.2f   %6.2f   %6.2f   %6.2f" % (
                    self.allMse['exp'], 
                    self.allMse['hyp'], 
                    self.allMse['lin'], 
                    self.allMse['log']
            )))
            print(("a        %6.2f   %6.2f   %6.2f   %6.2f" % (
                    self.allFitParams['exp'][0], 
                    self.allFitParams['hyp'][0],
                    self.allFitParams['lin'][0], 
                    self.allFitParams['log'][0]
            )))
            print(("b          -     %6.2f    -      %6.2f" % (
                    self.allFitParams['hyp'][1],
                    self.allFitParams['log'][1]
            )))
            print(("c           -         -       -      %5.2f" % (
                    self.allFitParams['log'][2]
            )))

    def save_(self, file='tree_growth.csv'):
        """Save growth stuff to a csv file
        Default path is in OUT_DIR.

        Args:
            file: name or path to csv file

        """
        # growth data
        io_.print_csv(
                file, np.column_stack((self.age, self.diam, self.biomass)),
                col_names=['age','diam','biomass',"allom="+self.allom]
        )
        
        # fitted data - in a list because of the size heterogeneity
        fit_file = file.split(".csv")[0] + "_fit.csv"
        data = np.column_stack(
                (self.biomass, self.allFitData['exp'], 
                 self.allFitData['hyp'], self.allFitData['lin'], 
                 self.allFitData['log'])
        )
        cols = ['data', 'exp', 'hyp', 'lin', 'log']
        io_.print_csv(fit_file, data, col_names=cols)
        
        # fit parameters
        param_file = file.split(".csv")[0] + "_fit_params.csv"
        temp = {'exp':[], 'hyp':[], 'lin':[], 'log':[]}
        row1 = []
        row2 = []
        row3 = []
        row4 = []
        for s in ['exp', 'hyp', 'lin', 'log']:
            row1.append(self.allMse[s])
            row2.append(self.allFitParams[s][0])
            try:
                row3.append(self.allFitParams[s][1])
            except IndexError:
                row3.append('')
            try:
                row4.append(self.allFitParams[s][2])
            except IndexError:
                row4.append('')
        
        data = [row1, row2, row3, row4]
        cols = ['exp', 'hyp', 'lin', 'log']
        io_.print_csv(param_file, data, col_names=cols)

        # biomass, and four fits
        # params for each fit

#--------------------------------------------------
# Begin methods for allometric equations
# NOTE: RETURNS kg C - WHEN USING WITH CARBON POOLS
# (TREE MODEL), MAKE SURE TO CONVERT TO t C
#--------------------------------------------------

# Return AGB of general allometric of type diam = -c*(agb)^c
#   -> solving for agb gives agb = exp[(ln(diam) - ln(c)) / d]
#   -> agb = exp[a + b*ln(diam)]    since c,d are arbitrary

def log_allom(params, d, dens=None):
    """General log type allometric.
    
    Args:
        params: array-like holding fit params
                with p[0] multiplied by highest power of ln(d)
                e.g. p=np.array([2.601, -3.629])
                     -> agb = exp(2.601*log(d)-3.629) (kg) is the ryan allom
        d: diameter
        dens: tree density (only needed for some allometrics)
    Returns:
        agb: agb corresponding to diameter of d
        
    """
    if math.fabs(d) < 0.00000001:
        agb = 0
    else:
        logd = math.log(d)
        agb = np.polyval(params, logd)
        agb = math.exp(agb)     # kg C
        if dens is not None:    # for Chave allometric (and possibly others)
            agb *= dens
    
    return agb

# Some specific log allometrics
# All take dbh vectors and tree object as arguments
def ryan(dbh, tree_params):
    """C. Ryan, biotropica (2010)."""
    return log_allom([2.601,-3.629], dbh)

def tumwebaze_grevillea(dbh, tree_params):
    """Tumwebaze et al. (2013) - Grevillea."""

    agb = log_allom([3.06,-5.5], dbh) + log_allom([1.32,1.06], dbh)
    return agb * tree_params.carbon

def tumwebaze_maesopsis(dbh, tree_params):
    """Tumwebaze et al. (2013) - Maesopsis."""

    agb = log_allom([3.33,-7.02], dbh) + log_allom([2.38,-2.9], dbh)
    return agb * tree_params.carbon    

def tumwebaze_markhamia(dbh, tree_params):
    """Tumwebaze et al. (2013) - Markhamia."""
    agb = log_allom([2.63,-4.91], dbh) + log_allom([2.43,-3.08], dbh)
    return agb * tree_params.carbon

def chave_dry(dbh, tree_params):
    """Chave et al. (2005) generic tropical tree 
    with < 1500 mm/year rainfall, > 5 months dry season
        
    """
    agb = log_allom([-0.0281,0.207,1.784,-0.667], dbh, dens=tree_params.dens)
    return agb * tree_params.carbon

def chave_moist(dbh, tree_params):
    """Chave et al. (2005) generic tropical tree
    with 1500-3000 mm/year rainfall, 1-4 months dry season
    
    """
    agb = log_allom([-0.0281,0.207,2.148,-1.499], dbh, dens=tree_params.dens)
    return agb * tree_params.carbon

def chave_wet(dbh, tree_params):
    """Chave et al. (2005) generic tropical tree
    with > 3500 mm/year rainfall, no seasonality
    
    """
    agb = log_allom([-0.0281,0.207,1.98,-1.239], dbh, dens=tree_params.dens)
    return agb * tree_params.carbon
    
allometric = {
        'ryan': ryan,
        'grevillea': tumwebaze_grevillea,
        'maesopsis': tumwebaze_maesopsis,
        'markhamia': tumwebaze_markhamia,
        'chave dry': chave_dry,
        'chave moist': chave_moist,
        'chave wet': chave_wet,
        'log_allom': log_allom
}
