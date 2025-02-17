#!/usr/bin/python

"""
Module containing soil model classes.
class RothC(object)         parent class
class InverseRothC(RothC)   inverse soil model (finding equil. distro.)
class ForwardRothC(RothC)   forward soil model (solve ODEs forward in time)

NOTE, all units are in t C ha^-1
"""

import sys
import os
import logging as log

import math
import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt

from ..common import csv_handler

from .. import configuration
from ..command_line import emit

    
class RothC(object):
    
    """
    Object for RothC soil models. 
    Includes methods and variables common to both forward and inverse.
     
    Instance variables
    ------------------
    k       rate constants for the 4 soil pools (with RMF)
    cover   vector with soil cover for each month (1 if covered, 0 else)

    """
    
    # Class variables (defaults)
    K_BASE = np.array([10.0, 0.3, 0.66, 0.02])
    
    def __init__(self, soil, climate, cover):
        """Initialise rothc object.
        
        Args:
            soil: SoilParams object with soil parameters
            climate: Climate object with climate parameters
            cover: monthly cover vector (1=covered, 0=uncovered)

        """
        self.soil = soil
        self.climate = climate
        self.cover = cover
        self.k = self.get_rmf() * RothC.K_BASE
    
    # Rate modifying-factor function - needed in forward and inverse
    def get_rmf(self):
        """Calculate the rate modifying factor b
        based on climate and soil cover.
        
        Returns:
            rmf: product of the all three RMFs (as a mean for the year)

        """
        
        # Calculation of b (topsoil moisture deficit RMF)
        # Deficit is difference between rain and evaporation (pet/0.75)
        deficit = self.climate.rain - self.climate.evap
        m = self._get_first_pos_def(deficit)
        m, rainAlwaysExceedsEvap = self._get_first_neg_def(deficit, m)
        b = np.ones(12)
        if rainAlwaysExceedsEvap:
            return b.mean()

        # Rainfall < evap in month m, so
        # tart calculating SMD from month before m
        m -= 1
        cc = self.soil.clay
        d = self.soil.depth
        max = -(20+ 1.3*cc - 0.01*(cc**2)) * (d/23.0)
        accTsmd = 0.0
        tsmd = np.zeros(12)

        # Now define deficit as rain - pet
        deficit = self.climate.rain - self.climate.evap*0.75
        
        # Loop through each month
        for i in range(12):
            accTsmd = self._get_acc_tsmd(
                    accTsmd, deficit[m], self.cover[m], max)
            if accTsmd >= 0.444*max:
                b[m] = 1
            elif accTsmd >= max:
                b[m] = 0.2 + 0.8*(max-accTsmd) / (0.556*max)
            else:
                log.error("DEFICIT = %5.2f" % accTsmd)
                sys.exit(1)

            tsmd[m] = accTsmd
            m += 1
            if m > 11:
                m = 0

        # Temperature RMF (a)
        a = np.zeros(12)
        for i in np.where(self.climate.temp > -5.0):
            a[i] = 47.91 / (1.0 + np.exp(106.06 / (self.climate.temp+18.27)))

        # Soil cover RMF (c)
        c = np.ones(12)
        for i in np.where(self.cover == 1)[0]:
            c[i] = 0.6

        return (a*b*c).mean()   # yearly average of total RMF

    # Helper methods for finding b (topsoil moisture RMF)
    # Find first month where deficit > 0
    def _get_first_pos_def(self, deficit):
        isSane = False
        m = 0
        for i in np.where(deficit > 0):
            if any(i):   # could be empty list
                isSane = True
                m = min(i)
                break
        
        if not isSane:
            log.warning("EVAPORATION ALWAYS EXCEED RAINFALL")
            m = 0

        return m  # first month where deficit > 0
    
    # Find first month after m where rainfall < evap (deficit<0)
    def _get_first_neg_def(self, deficit, m):
        rainAlwaysExceedsEvap = True
        for i in range(12):
            m += 1
            if m > 11:
                m = 0
            if deficit[m] < 0:
                rainAlwaysExceedsEvap = False
                break

        return m, rainAlwaysExceedsEvap

    # Get accTSMD for a given month
    def _get_acc_tsmd(self, smd, def_m, cover_m, max):
        if def_m > 0:
            # Add excess rain to SMD
            smd += def_m
            if smd > 0:
                smd = 0
        else:
            # deficit < 0
            if cover_m == 1:
                # Crop present, so increase SMD
                smd += def_m
                if smd < max:
                    smd = max
            else: 
                # Crop not present
                if smd < 0.556*max:
                    pass
                else:
                    # Increase SMD
                    smd += def_m
                    if smd < 0.556*max:
                        smd = 0.556*max
        # End if-else for deficit > 0
        return smd

    def dC_dt(self, C, t, x, k, input):
        """Function for system of differential equations governing
        amounts of carbon in each pool (vector C).
        
        Args:
            C: vector of carbon pools
            t: time (not used, but needed for the scipy.optimize fit)
            x: partitioning coefficient
            k: rate constants
            input: soil input in the given year
        Returns:
            rhs: array of the RHS of dC/dt.

        """

        # carbon gain from decay (goes to BIO, HUM, CO2)
        bioHumIn = C[0]*k[0] + C[1]*k[1] + C[2]*k[2] + C[3]*k[3]

        rhs = np.array([
                input*x[0] - C[0]*k[0],
                input*x[1] - C[1]*k[1],
                bioHumIn*x[2] - C[2]*k[2],
                bioHumIn*x[3] - C[3]*k[3]
        ])

        return rhs
    

class InverseRothC(RothC):

    """
    Inverse RothC model. Extends RothC class.
    
    Instance variables
    ------------------
    eqC     calculated equilibrium distribution of carbon
    inputC  yearly input to soil giving eqC
    x       partitioning coefficients

    """
    
    def __init__(self, soil, climate, cover=np.ones(12)):
        """Initialise inverse rothc object.
        
        Args:
            soil: soil object with soil parameters
            climate: climate object with climate parameters
            cover: monthly soil cover vector

        """
        super(InverseRothC, self).__init__(soil, climate, cover)
        self.eqC, self.inputC, self.x = self.solver()
        
    def solver(self):
        """Run RothC in 'inverse' mode to find inputs 
        giving equilibrium and Carbon pool values at equilibrium. 

        Returns:
            eqC: equilibrium distribution of carbon
            eqInput: yearly tree input giving equilibrium
            x: partitioning coefficient for pools

        """

        # Partioning coefficients 
        x = self.get_partitions()
        t = 0   # just needed to define for dC_dt function
        C0 = np.array([0.1, 10, 0, 0])  # initial ballpark guess

        # Keep track of difference b/t Ctot and Ceq
        prevDiff = 1000.0

        # Loop through range of inputs
        for input in np.arange(0.01, 10, 0.001):
            C = optimize.fsolve(self.dC_dt, C0, args=(t, x, self.k, input))
            # TODO: Is this correct
            Ctot = np.sum(np.array(C)) + self.soil.iom
            currDiff = math.fabs(Ctot - self.soil.Ceq)

            if currDiff < prevDiff:
                prevDiff = currDiff
                inputC = input
                prevC = C
                prevCtot = Ctot

            if prevDiff < currDiff:
                break
        
        eqC = prevC
        eqInput = inputC
        return eqC, eqInput, x

    def get_partitions(self):
        """Calculate partitioning coefficients.
        
        Returns:
            x: partitioning coefficient for the 4 pools

        """
        
        # Determine p_2 (fraction of input going to CO2) based on clay content
        z = 1.67 * (1.85 + 1.6*math.exp(-0.0786 * self.soil.clay))
        p2 = z/(z+1)

        # p_3 is always 0.46 (see RothC papaer
        p3 = 0.46

        # p_1 is dpm fraction of input:
        #   deciduous tropical woodland: dpm=0.2, rpm=0.8
        #   crops: dpm=0.59, rpm=0.41

        # no crops at equil so p1 always 0.2
        p1 = 0.2
        return np.array([
                p1,
                1-p1,
                p3*(1-p2),
                (1-p2)*(1-p3)
        ])

    def print_(self):
        """Print data from inverse RothC run to stdout."""

        pools = ['DPM', 'RPM', 'BIO', 'HUM']
        print ("\nINVERSE CALCULATIONS")
        print ("====================\n")
        print("Equilibrium C -",self.eqC.sum() + self.soil.iom)
        for i in range(len(self.eqC)):
            print("   ",pools[i],"- - - -",self.eqC[i])
        print("    IOM","- - - -",self.soil.iom)
        print("Equil. inputs -",self.inputC)
        print ("")

    def save_(self, file='soil_model_inverse.csv'):
        """Save data to csv. Default path is OUTPUT_DIR."""
        
        data = np.array([
                self.eqC.sum()+self.soil.iom, self.eqC[0], self.eqC[1], 
                self.eqC[2], self.eqC[3], self.soil.iom, self.inputC])
        cols = ['Ceq', 'dpm', 'rpm', 'bio', 'hum', 'iom', 'inputs']
        csv_handler.print_csv(file, data, col_names=cols)


class ForwardRothC(RothC):
    
    """
    Forward RothC class. Extends RothC class.

    Instance variables
    ----------------
    SOC     vector with soil distributions for each year

    """
    
    def __init__(
            self, soil, climate, cover,
            Ci, crop=[], tree=[], litter=[], fire =[],
            solveToValue=False):
        """Initialise ForwardRothC.

        Args:
            soil: soil object
            climate: climate object
            cover: soil cover vector
            Ci: initial carbon pools (for solver)
            crop: list of crop objects which provide carbon to soil
            tree: list of tree objects which provide carbon to soil
            litter: list of litter objects which provide carbon to soil
            solveToValue: whether to solve to value (to Cy0) or by time
        
        """
        super(ForwardRothC, self).__init__(soil, climate, cover)
        self.SOC,self.inputs,self.Cy0Year = self.solver(
                Ci, crop, tree, litter, fire, solveToValue) 
    
    def solver(
            self, Ci,
            crop=[], tree=[], litter=[], fire =[],
            solveToValue=False):
        """ Run RothC in 'forward' mode; 
        solve dC_dt over a given time period
        or to a certain value, given a vector with soil inputs. 
        Use scipy.integrate.odeint as a Runge-Kutta solver.
        
        Args: 
            crop: list of Crop objects (not reduced by fire)
            tree: list of Tree objects (not reduced by fire)
            solveToValue: whether to solve to a particular value (Cy0)
                          as opposed to for a certain amount of time
        Returns:
            C: vector with yearly distribution of soil carbon
            inputs: yearly inputs to soil with 2 columns (crop,tree)
            year_target_reached: year that the target value
                    (if solveToValue==True) of Cy0 was reached
                    since it will (probably) not be on nice round num.
        
        """
        
        # Reduce inputs due to fire
        soilIn_crop,soilIn_tree = emit.reduceFromFire(crop, tree, litter, fire)
        
        # make input into array with 2 columns (soilIn_crop,soilIn_tree)
        inputs = np.column_stack((soilIn_crop, soilIn_tree))

        # Calculate yearly values of x based dpm:rpm ratio for each year
        x = self.get_partitions(inputs)
        t = np.arange(0, 1, 0.001)
        C = np.zeros((configuration.N_YEARS+1, 4))
        C[0] = Ci
        Ctot = np.zeros(configuration.N_YEARS+1)
        
        # keep track of when target year reached
        year_target_reached = 0
        # To keep track of closest value to SOCf
        if solveToValue:
            prevDiff_outer = 1000.0
            prevDiff_inner = 1000.0
        

        for i in range(1,configuration.N_YEARS+1):
            # Careful with indices - using 1-based for arrays here
            # since, e.g., C[2] should correspond to carbon after 2 years


            # Solve the diffEQs to get pools for year i
            Ctemp = integrate.odeint(
                    self.dC_dt, C[i-1], t,
                    args=(x[i-1], self.k, inputs[i-1].sum())
            )
            C[i] = Ctemp[-1]    # carbon pools at end of year
        
            # Check to see if close to target value
            if solveToValue:
                j = -1
                for c in Ctemp:
                    j += 1
                    Ctot = c.sum() + self.soil.iom
                    currDiff = math.fabs(Ctot - self.soil.Cy0)

                    if currDiff - prevDiff_inner < 0.00000001: 
                        prevDiff_inner = currDiff
                        c_inner = c
                        closest_j = j
                    else: # getting farther away 
                        break

                if prevDiff_inner < prevDiff_outer:
                    prevDiff_outer = prevDiff_inner
                    c_outer = c_inner
                    closest_i = i
                else: 
                    break
        
        if solveToValue:
            C[closest_i] = c_outer
            C = C[0:closest_i+1]
            year_target_reached = float(closest_j) / len(t)
            inputs = inputs[0:len(C[:,0])-1]
        
        return C,inputs,year_target_reached
    
    def get_partitions(self, inputs):
        """Calculate partitioning coefficients.

        Args:
            input: 2d array with crop_in,tree_in inputs as the columns
        Returns:
            x: partitioning coefficient (as a vector for each year)
        
        """
        
        # Determine p_2 (fraction of input going to CO2) based on clay content
        z = 1.67 * (1.85 + 1.6*math.exp(-0.0786*self.soil.clay))
        p2 = z/(z+1)

        # p_3 is always 0.46 (see RothC papaer
        p3 = 0.46

        # p_1 is dpm fraction of input:
        #   deciduous tropical woodland: dpm=0.2, rpm=0.8
        #   crops: dpm=0.59, rpm=0.41
        # So weigh p_1 according to amount of trees/crops for each year
        
        # Normalize
        i = 0
        normInput = np.zeros((configuration.N_YEARS,2))
        for row in inputs:
            if math.fabs(float(row.sum())) < 0.00000001:
                normInput[i] = 0
            else:
                normInput[i] = inputs[i] / float(row.sum())
            i += 1
        
        # Weighted mean of dpm
        p1 = np.array(0.59*normInput[:,0] + 0.2*normInput[:,1])

        # Construct x
        # make arrays (p1 already array)
        x = np.column_stack(
                (p1, 1-p1, 
                 np.array(configuration.N_YEARS * [p3*(1-p2)]), 
                 np.array(configuration.N_YEARS * [(1-p2)*(1-p3)]))
        )

        return x

    def plot_(self, legendStr, saveName=None):
        """Plot total carbon vs year for forwardRothC run.
        
        Args:
            legendStr: string to put in legend

        """
        try:
            # figure has been initialized before
            ax = ForwardRothC.ax
        except AttributeError:
            # first time this has been called
            fig = plt.figure()
            fig.suptitle("Soil Carbon")  # Replace set_window_title with suptitle
            ForwardRothC.ax = fig.add_subplot(1,1,1)
            ax = ForwardRothC.ax
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("SOC (t C ha^-1)")
            ax.set_title("Total soil carbon vs time")
    
        tot_soc = np.sum(self.SOC, axis=1)
        if len(tot_soc) == configuration.N_YEARS+1:
            # baseline or project
            x = list(range(len(tot_soc)))
        else:
            # initialisation run is before year 0
            x = np.array(list(range(-len(tot_soc)+2, 2)))
            x = x - self.Cy0Year
            x[-1] = 0

        tot_soc = np.sum(self.SOC, axis=1)
        ax.plot(x, tot_soc, label=legendStr)
        ax.legend(loc='best')
        
        if saveName is not None:
            plt.savefig(os.path.join(configuration.OUTPUT_DIR, saveName))

    def print_(self, solveToValue=False):
        """Print data from forward RothC run to stdout."""
        print ("\n\nFORWARD CALCULATIONS")
        print ("====================\n")
        print("Length: ", configuration.N_YEARS, "years")
        print ("year carbon  crop_in  tree_in")
        tot_soc = np.sum(self.SOC, axis=1)
        if len(tot_soc) == configuration.N_YEARS+1:
            for i in range(len(tot_soc)):
                if i == configuration.N_YEARS:
                    print(i, "  ",)
                    print(tot_soc[i] + self.soil.iom)
                else:
                    print(i, "  ", )
                    print(tot_soc[i] + self.soil.iom, "  ",)
                    print(self.inputs[i][0], "  ", self.inputs[i][1])
        else:
            x = np.array(list(range(-len(tot_soc)+2,2)))
            x = x - self.Cy0Year
            x[-1] = 0
            for i in range(len(tot_soc)):
                print("%6.3f" % (x[i]),)
                print("  ",)
                print(tot_soc[i] + self.soil.iom, "  ",)
                if i == len(tot_soc)-1:
                    pass
                else:  
                    print(self.inputs[i][0], "  ",self.inputs[i][1])

    def save_(self, file='soil_model_forward.csv'):
        """Save data from forward RothC run to a csv.
        Default path is OUTPUT_DIR.

        """
        tot_soc = np.sum(self.SOC, axis=1)
        inputs = np.append(self.inputs, [[0,0]], axis=0)
        data = np.column_stack(
                (tot_soc+self.soil.iom, self.SOC, 
                 np.array(len(tot_soc)*[self.soil.iom]),
                 inputs[:,0], inputs[:,1])
                )
        cols = [
                'soc', 'dpm', 'rpm', 'bio', 'hum', 
                'iom', 'crop_in', 'tree_in'
        ]
        if len(tot_soc) != configuration.N_YEARS+1:   # solve to value
            cols.insert(0, 'year')
            x = np.array(list(range(-len(tot_soc)+2,2)))
            x = x - self.Cy0Year
            x[-1] = 0
            data = np.column_stack((x,data))
            csv_handler.print_csv(file, data, col_names=cols)
        else:
            csv_handler.print_csv(file, data, col_names=cols, print_years=True)


