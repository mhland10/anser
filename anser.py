"""
ANSER

Author:     Matthew Holland

This module is the Python software that allows for the reading and post-processing of CFD.

"""

import os
import sys
import numpy as np

###################################################################################################
#
# Add supporting directories to path
#
###################################################################################################

anser_dir = os.path.dirname( os.path.abspath(__file__) )
lib_path = "lib"
img_path = "img"

lib_dir = os.path.join( anser_dir , lib_path )
img_dir = os.path.join( anser_dir , img_path )

sys.path.insert( 0 , lib_dir )
sys.path.insert( 0 , img_dir )

from distributedObjects import *
from distributedFunctions import *

###################################################################################################
#
# Pre-processing objects
#
###################################################################################################

class syntheticBoundaryLayer:

    def __init__( self , vonKarmanConst = 0.41 , vanDriestConst = 5.0 , distDomainLims = [ 1e-3 , 1e3 ] , distDomainN = 1e3 ):
        """
        This object is the data object that contains the data and methods to create a synthetic 
            boundary layer.

        INPUTS
        ------

        **vonKarmanConst : <float>  The von Karman constant that will be used to define the law-of-
                                        the-wall distribution.

        **vanDriestConst : <float>  The constant that define the van Driest profile of the law-of-
                                        the-wall distribution.

        **distDomainLims : [float]  The limits of the tabulated law-of-the-wall distribution's 
                                        domain. 

        **distDomainN : <int>   The number of points in the tabulated law-of-the-wall
                                    distribution's domain.

        ATTRIBUTES
        ----------

        profile :   The boundaryLayer object that defines the non-dimensional boundary layer.
 
        """

        self.profile = boundaryLayer( vonKarmanConst = vonKarmanConst , vanDriestConst = vanDriestConst , distDomainLims = distDomainLims , distDomainN = distDomainN )

    def dimensionalize( cls , U_inf , nu , C_f = None , u_tau = None ):
        """

        This method takes the profile and applies the parameters to calculate the boundary layer.

        A note on units:    The method is written for SI. If other units are to be used, the unit
                                system must follow the same convention.

        Args:
            U_inf (float):  The freestream velocity of the flow outside the boundary layer.

            nu (float):     Molecular kinematic viscosity.

            C_f (float, optional):  Skin friction coefficient of the boundary layer. Defaults to 
                                        None. Must be numerical if "u_tau" is None.
            
            u_tau (float, optional):    Friction velocity of the boundary layer. Defaults to 
                                            None. Must be numerical if "C_f" is None.
        
        """

        #
        # Set up the variables
        #
        cls.U_inf = U_inf
        cls.nu = nu

        if C_f:
            cls.C_f = C_f
            cls.u_tau = cls.U_inf * np.sqrt( cls.C_f / 2 )
        elif u_tau:
            cls.u_tau = u_tau
            cls.C_f = 2 * ( ( cls.u_tau / cls.U_inf ) ** 2 )
        else:
            raise ValueError( "C_f or u_tau are missing" )
        
        #
        # Turn values to dimensional values
        #
        cls.Us = cls.profile.Upluss * cls.u_tau
        cls.ys = cls.profile.ypluss * ( cls.nu / cls.u_tau )
        cls.u_U = cls.Us / cls.U_inf

    def boundaryLayerLimits( cls , delta = None , delta_star = None , theta = None , BL_threshold = 0.99 ):

        #
        # Find the edge of the boundary layer
        #
        cls.Uplus_edge = cls.U_inf * BL_threshold / cls.u_tau
        cls.yplus_edge = np.interp( cls.Uplus_edge , cls.profile.Upluss , cls.profile.ypluss )
        cls.profile.Upluss[cls.profile.ypluss>=cls.yplus_edge] = cls.Uplus_edge
        cls.delta = cls.yplus_edge * ( cls.nu / cls.u_tau )

        #
        # Calculate boundary layer height
        #
        cls.u_U = (( cls.profile.Upluss *  cls.u_tau ) / cls.U_inf )
        cls.ys = cls.profile.ypluss * ( cls.nu / cls.u_tau )
        cls.y_delta = cls.ys #/ cls.delta
        cls.delta_star = np.trapz( 1 - cls.u_U , x = cls.y_delta )
        cls.theta = np.trapz( cls.u_U * ( 1 - cls.u_U ) , x = cls.y_delta )
        

class recycledBoundaryLayer:
    """
    This object finds the boundary layer location desired and produces a mapped boundary layer to
        inject in a new inlet.
    
    """

    def __init__( self , bounding_points , rake_length , boundaryLayer_data , nu , datafile , N_samples = 100 , scanaxis = 'x' , norm_vector = (0,-1,0) , data_weights = None , start_height = 1e-6 ):
        """
        Initialize the recycling boundary layer

        Args:
            bounding_points ([float]):  These are the points at either end of the domain to sweep
                                            the rake across to find the best boundary layer 
                                            location. Must align with "scanaxis".

            rake_length (float):    The length of the rakes that will find the best boundary layer
                                        location.

            boundaryLayer_data [float]: This is a list or array of the data to fit the sampling
                                            that the recycling will be fit to. This will take the
                                            form of a normalized error to be minimized depending on
                                            the location. The order needs to be:

                                        1)  Boundary Layer Height (delta) [m]

                                        2)  Displacement Boundary Layer Height (delta*) [m]

                                        3)  Momentum Boundary Layer Height (theta) [m]

                                        4)  Shape Factor (H) [-]

                                        5)  Skin Friction Coefficient (C_f) [-]

                                        6)  Shear Velocity (u_tau) [m/s]

                                        7)  Unit Length Reynolds Number (dRe_x/dx) [1/m]

                                        8)  Momentum Thickness Reynolds Number (Re_theta) [-]

                                        9)  Shear Velocity Reynolds Number (Re_tau) [-]

                                        If a value is not necessary, then the corresponding weight
                                            in "data_weights" should be 0. 

                                        The net error is calculated by: norm( ( weight * error) )

                                        The error is calculated by: ( station / desired ) - 1

            nu (float): The kinematic viscosity of the CFD conditions.

            datafile (string):  The data file to pull the boundary layer from.
            
            N_samples   (float, optional):  The number of samples that will be in the rakes that
                                                are sampling the boundary layer to find where best
                                                represents the expected boundary layer.

            scanaxis    (char, optional): The axis to scan along to find the best boundary layer 
                                            representation. Defaults to 'x'.

            data_weights [float, optional]: The weights that correspond to the data in 
                                                "boundaryLayer_data". Defaults to None, which is an
                                                equal weighting to all data values.

            start_height    (float, optional):  The starting height for the rake. Needs to be non-
                                                    zero for the logarithmic distribution.

        """

        #
        # Sweep Parameters
        #
        self.bounding_points = bounding_points
        self.rake_length = rake_length
        self.N_samples = N_samples
        self.scanaxis = scanaxis
        self.start_height = start_height

        #
        # Boundary Layer Data
        #
        self.boundaryLayer_data = boundaryLayer_data
        self.data_weights = data_weights

        self.datafile = datafile

        self.nu = nu

    def bisect_Search( cls ):

        searching = True
        c = 1
        search_bounds = cls.bounding_points
        while searching:
            print("Search iteration:\t{x}".format(x=c))

            if c==1:
                boundLHS = search_bounds[0]
                boundRHS = search_bounds[1]
            print("\tLHS:\t{x:.3f}".format(x=boundLHS))
            print("\tRHS:\t{x:.3f}".format(x=boundRHS))

            #
            # Get data for LHS & RHS
            # 
            cls.rakeLHS = rake( ( boundLHS * np.ones( cls.N_samples ) , -np.logspace( np.log10( cls.start_height ) , np.log10( cls.rake_length - cls.start_height ) , num = cls.N_samples ) , np.zeros( cls.N_samples ) ) , cls.datafile )
            cls.rakeRHS = rake( ( boundRHS * np.ones( cls.N_samples ) , -np.logspace( np.log10( cls.start_height ) , np.log10( cls.rake_length - cls.start_height ) , num = cls.N_samples ) , np.zeros( cls.N_samples ) ) , cls.datafile )
            cls.rakeLHS.dataToDictionary()
            cls.rakeRHS.dataToDictionary()

            #
            # Calculate flow parameters
            #
            cls.LHS_data = [0]*3
            cls.RHS_data = [0]*3
            # BL thicknesses
            cls.LHS_data[0] , cls.LHS_data[1] , cls.LHS_data[2] = boundaryLayerThickness( np.abs( cls.rakeLHS.data['y'] ) , cls.rakeLHS.data['U'][:,0] )
            cls.RHS_data[0] , cls.RHS_data[1] , cls.RHS_data[2] = boundaryLayerThickness( np.abs( cls.rakeRHS.data['y'] ) , cls.rakeRHS.data['U'][:,0] )
            # Shape factor
            cls.LHS_data += [ cls.LHS_data[1] / cls.LHS_data[2] ]
            cls.RHS_data += [ cls.RHS_data[1] / cls.RHS_data[2] ]
            # Shear parameters
            cls.LHS_data += [0]*2
            cls.RHS_data += [0]*2
            cls.LHS_data[-2] , cls.LHS_data[-1] = shearConditions( np.abs( cls.rakeLHS.data['y'] ) , cls.rakeLHS.data['U'][:,0] , cls.nu )
            cls.RHS_data[-2] , cls.RHS_data[-1] = shearConditions( np.abs( cls.rakeRHS.data['y'] ) , cls.rakeRHS.data['U'][:,0] , cls.nu )
            # Reynolds numbers
            cls.LHS_data += [0]*3
            cls.RHS_data += [0]*3
            cls.LHS_data[-3] = ReynoldsNumber( 0 , cls.nu , u = cls.rakeLHS.data['U'][:,0] )
            cls.RHS_data[-3] = ReynoldsNumber( 0 , cls.nu , u = cls.rakeRHS.data['U'][:,0] )
            cls.LHS_data[-2] = ReynoldsNumber( cls.LHS_data[2] , cls.nu , u = cls.rakeLHS.data['U'][:,0] )
            cls.RHS_data[-2] = ReynoldsNumber( cls.RHS_data[2] , cls.nu , u = cls.rakeRHS.data['U'][:,0] )
            cls.LHS_data[-1] = ReynoldsNumber( cls.LHS_data[0] , cls.nu , U_inf = cls.LHS_data[3] )
            cls.RHS_data[-1] = ReynoldsNumber( cls.RHS_data[0] , cls.nu , U_inf = cls.RHS_data[3] )

            #
            # Calculate normalized error
            #
            cls.LHS_errors = np.asarray( cls.LHS_data ) / cls.boundaryLayer_data - 1
            cls.RHS_errors = np.asarray( cls.RHS_data ) / cls.boundaryLayer_data - 1
            cls.LHS_errorNorm = np.linalg.norm( cls.LHS_errors )
            cls.RHS_errorNorm = np.linalg.norm( cls.RHS_errors )
            print("\tLHS norm:\t{x:.3f}".format(x=cls.LHS_errorNorm))
            print("\tRHS norm:\t{x:.3f}".format(x=cls.RHS_errorNorm))

            # 
            # Bisect
            #
            cut = np.mean([ boundLHS , boundRHS ])
            if cls.LHS_errorNorm > cls.RHS_errorNorm:
                LHS_nextbound = cut
                RHS_nextbound = boundRHS
                c += 1
            elif cls.LHS_errorNorm < cls.RHS_errorNorm:
                LHS_nextbound = boundLHS
                RHS_nextbound = cut
                c +=1
            elif c>10:
                searching = False
            else:
                searching = False


            #searching = False





        




        

