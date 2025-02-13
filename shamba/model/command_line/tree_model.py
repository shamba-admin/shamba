#!/usr/bin/python

"""Module containing Tree class."""

import os
import sys
import logging as log

import numpy as np
import math
import matplotlib.pyplot as plt

from ..common import csv_handler

from .. import configuration
from .tree_params import TreeParams


class TreeModel(object):

    """
    Tree model class. Calculate residues and soil inputs
    for given params and growth.
    
    Instance variables
    ----------------
    tree_params     TreeParams object with the params (dens, carbon, etc.)
    growth          TreeGrowth object governing growth of trees
    output          output to soil,fire in t C ha^-1
                        (dict with keys 'carbon,'nitrogen','DMon','DMoff')
    woodyBiom       vector of woody biomass in each pool in t C ha^-1

    """

    def __init__(
            self, tree_params, tree_growth, pool_params, 
            yearPlanted=0, initialStandDens=0, # Should initialStandDens be 200?
            thin=None, mort=None):
        """Intialise TreeModel object (run biomass model, essentially).
        
        Args:
            tree_params TreeParams object (holds tree params)
            growth: tree_growth.Growth object
            pool_params: dict with alloc, turnover, thinFrac, and mortFrac
            yearPlanted: year (after start of project) tree is planted
            thin: thinning vector
            thinFrac: vector with the retained fraction of thinned biomass
                      for each pool (default = [leaf=1,branch=0,stem=0,
                                                croot=1,froot=1])
            mort: mortality vector
            mortFrac: vector with the retained fraction of dead biomass
                      for each pool (default same as thinFrac)
        Raises:
            KeyError: if pool_params doesn't have the appropriate keys

        """

        self.tree_params = tree_params
        self.tree_growth = tree_growth
        try:
            self.alloc  = pool_params['alloc']
            self.turnover = pool_params['turnover']
            self.thinFrac = pool_params['thinFrac']
            self.mortFrac = pool_params['mortFrac']
            if thin is None:
                self.thin = np.zeros(configuration.N_YEARS+1)
            else:
                self.thin = thin 
            if mort is None:
                self.mort = np.zeros(configuration.N_YEARS+1)
            else:
                self.mort = mort
            
        except KeyError:
            log.exception("Pool parameters not provided.")
            sys.exit(1)
         
        initialBiomass = self.tree_growth.fitData[0]
        
        self.output,self.woodyBiom,self.balance = self.get_inputs(
                initialBiomass, yearPlanted, initialStandDens)
    
    @classmethod
    def from_defaults(
            cls, tree_params, tree_growth, 
            yearPlanted=0, standDens=100, # Should standDens be 200?
            thin=None, thinFrac=None,
            mort=None, mortFrac=None):
        """Use defaults for pool params.
        Can override defaults for thinFrac and mortFrac by providing arguments.
        
        """

        data = csv_handler.read_csv('biomass_pool_params.csv', cols=(3,4,5,6))
        turnover = data[:,0]
        alloc = data[:,1]
        thinFrac_temp = data[:,2]
        mortFrac_temp = data[:,3]
        
        # Take into account croot alloc - rs * stem alloc
        alloc[3] = alloc[2] * tree_params.rootToShoot

        # thinning and mortality
        if thin is None:
            thin = np.zeros(configuration.N_YEARS+1)
        if thinFrac is None:
            thinFrac = thinFrac_temp
        if mort is None:
            mort = np.zeros(configuration.N_YEARS+1)
        if mortFrac is None:
            mortFrac = mortFrac_temp
        
        params = {
                'alloc': alloc,
                'turnover': turnover,
                'thinFrac': thinFrac,
                'mortFrac': mortFrac
        }
        return cls(
                tree_params, tree_growth, params, 
                yearPlanted, standDens, thin, mort)

    def get_inputs(
            self, initialBiomass, yearPlanted, initialStandDens):
        """
        Calculate and return residues and soil inputs from the tree.

        Args:
            same explanations as __init__
        Returns:
            output: dict with soil,fire inputs due to this tree 
                         (keys='carbon','nitrogen','DMon','DMoff')
            woodyBiom: vector with yearly woody biomass pools
        
        **NOTE** a lot of these arrays are implicitly 2d with 2nd
        dimension = [leaf, branch, stem, croot, froot]. Careful here.

        """ 
        # NOTE - initially quantities are in kg C
        #   -> woodyBiom and output get converted at end before returning

        # First get params from bpFile and ppFile
        print ('new tree cohort running...')
        print(initialBiomass, yearPlanted, initialStandDens   )
        
        yp = yearPlanted
        print ('yp,  then initialSD')
        print (yp)
        print (initialStandDens )
        
        standDens = np.zeros(configuration.N_YEARS+1) 
        standDens[yp] = initialStandDens
        print ('standDens:')
        print (standDens)

        inputParams = {
                'live': np.array(
                                (configuration.N_YEARS+1) * [self.turnover]),
                'thin': self.thin,
                'dead': self.mort
        }
        retainedFrac = {
                'live': 1,
                'thin': self.thinFrac,
                'dead': self.mortFrac
        }
        
        # initialise stuff
        pools = np.zeros((configuration.N_YEARS+1,5))
        woodyBiom = np.zeros((configuration.N_YEARS+1,5))
        tNPP = np.zeros(configuration.N_YEARS+1)

        flux = {}
        inputs = {}
        exports = {}
        for s in ['live', 'dead', 'thin']:
            flux[s] = np.zeros((configuration.N_YEARS+1,5))
            inputs[s] = np.zeros((configuration.N_YEARS+1,5))
            exports[s] = np.zeros((configuration.N_YEARS+1,5))
        
        inputC = np.zeros((configuration.N_YEARS+1,5))
        exportC = np.zeros((configuration.N_YEARS+1,5))
        biomGrowth = np.zeros((configuration.N_YEARS+1,5))
        
        # set woodyBiom[0] to initial (allocated appropriately)
        pools[yp] = initialBiomass * self.alloc
        woodyBiom[yp] = pools[yp] * standDens[yp]
        for s in inputs:
            flux[s][yp] = woodyBiom[yp] * inputParams[s][yp]    

        in_ = np.zeros(configuration.N_YEARS+1)
        acc = np.zeros(configuration.N_YEARS+1)
        bal = np.zeros(configuration.N_YEARS+1)
        out = np.zeros(configuration.N_YEARS+1)
       
        for i in range(1+yearPlanted,configuration.N_YEARS+1):
            # Careful with indices - using 1-based here
            #   since, e.g., woodyBiom[2] should correspond to 
            #   biomass after 2 years
            
            agb = pools[i-1][1] + pools[i-1][2]
            
            # Growth for one tree
            tNPP[i] = self.tree_growth.funcDeriv[self.tree_growth.best](agb)
            biomGrowth[i] = tNPP[i] * self.alloc * standDens[i-1]
            
            for s in inputs:
                flux[s][i] = inputParams[s][i] * pools[i-1] * \
                            standDens[i-1]
                inputs[s][i] = retainedFrac[s] * flux[s][i]
                exports[s][i] = (1-retainedFrac[s]) * flux[s][i]

            # Totals (in t C / ha)
            inputC[i] = sum(inputs.values())[i]
            exportC[i] = sum(exports.values())[i]
            
            woodyBiom[i] = woodyBiom[i-1] 
            woodyBiom[i] += biomGrowth[i]
            woodyBiom[i] -= sum(flux.values())[i]

            standDens[i] = standDens[i-1]
            standDens[i] *= 1 - (
                    inputParams['dead'][i] + inputParams['thin'][i]
            )
            if standDens[i] < 1:
                print ('SD [i] is less than 1, end of this tree cohort...')
                break
            pools[i] = woodyBiom[i] / standDens[i]

            # Balance stuff
        
            in_[i] = biomGrowth[i].sum()
            acc[i] = woodyBiom[i].sum() - woodyBiom[i-1].sum()
            out[i]= inputC[i].sum() + exportC[i].sum()
            bal[i] = in_[i] - out[i] - acc[i]
        
        # ********************* 
        # Standard output stuff
        # in tonnes
        # *********************
        massBalance = {
                'in': in_ * 0.001,
                'out': out * 0.001,
                'acc': acc * 0.001,
                'bal': bal * 0.001
        }
        woodyBiom *= 0.001  # convert to tonnes for emissions calc.
        C = inputC[0:configuration.N_YEARS]
        DM = C / self.tree_params.carbon
        N = np.zeros((configuration.N_YEARS,5))
        for i in range(configuration.N_YEARS):
            N[i] = self.tree_params.nitrogen[0:configuration.N_YEARS] * DM[i]
        
        output = {}
        output['above'] = {
                'carbon': 0.001 * (C[:,0]+C[:,1]+C[:,2]),
                'nitrogen': 0.001 * (N[:,0]+N[:,1]+N[:,2]),
                'DMon': 0.001 * (DM[:,0]+DM[:,1]+DM[:,2]),
                'DMoff': np.zeros(len(C[:,0]))
        }
        output['below'] = {
                'carbon': 0.001*TreeParams.ROOT_IN_TOP_30*(C[:,3]+C[:,4]),
                'nitrogen': 0.001*TreeParams.ROOT_IN_TOP_30*(N[:,3]+N[:,4]),
                'DMon': 0.001*TreeParams.ROOT_IN_TOP_30*(DM[:,3]+DM[:,4]),
                'DMoff': np.zeros(len(C[:,0]))
        }
        return output,woodyBiom,massBalance

    def plot_biomass(self, saveName=None):
        """Plot the biomass pool data."""

        fig = plt.figure()
        fig.canvas.manager.set_window_title("Biomass Pools")
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.woodyBiom[:,0], label='leaf')
        ax.plot(self.woodyBiom[:,1], label='branch')
        ax.plot(self.woodyBiom[:,2], label='stem')
        ax.plot(self.woodyBiom[:,3], label='coarse root')
        ax.plot(self.woodyBiom[:,4], label='fine root')
        ax.legend(loc='best')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Pool biomass (t C ha^-1)')
        ax.set_title('Biomass pools vs time')
        
        if saveName is not None:
            plt.savefig(os.path.join(configuration.OUTPUT_DIR, saveName))

    def plot_balance(self, saveName=None):
        """Plot the mass balance data."""

        fig = plt.figure()
        fig.canvas.manager.set_window_title("Biomass Mass Balance")
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.balance['in'], label='in')
        ax.plot(self.balance['out'], label='out')
        ax.plot(self.balance['acc'], label='accumulated')
        ax.plot(self.balance['bal'], label='balance')
        ax.legend(loc='best')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Biomass (t C ha^-1)')
        ax.set_title('Mass balance vs time')
        
        if saveName is not None:
            plt.savefig(os.path.join(configuration.OUTPUT_DIR, saveName))

    def print_biomass(self):
        print ("\n\nBIOMASS MODEL")
        print ("=============\n")
        totalBiomass = np.sum(self.woodyBiom, axis=1)
        print ("year  biomass")
        for i in range(len(totalBiomass)):
            print(i, "  ",totalBiomass[i] )

    def print_balance(self):
        print(("\nMass-balance sum (kg C /ha): ", self.balance['bal'].sum()))
        totDiff = self.balance['bal'].sum() / self.woodyBiom[-1].sum()
        print(("Normalized mass balance (kg C /ha): ",totDiff))

    def save_(self, file='tree_model.csv'):
        """Save output and biomass to a csv file.
        Default path is in OUTPUT_DIR.

        Args:
            file: name or path to csv

        """
        # outputs
        cols = []
        data = []
        for s1 in ['above', 'below']: 
            for s2 in ['carbon', 'nitrogen', 'DMon', 'DMoff']:
                cols.append(s2+"_"+s1)
                data.append(self.output[s1][s2])
        data = np.column_stack(tuple(data))
        csv_handler.print_csv(file, data, col_names=cols)
        
        # biomass
        biomass_file = file.split(".csv")[0] + "_biomass.csv"
        cols = ['leaf', 'branch', 'stem', 'croot', 'froot']
        csv_handler.print_csv(
                biomass_file, self.woodyBiom, 
                col_names=cols, print_total=True, print_years=True)



