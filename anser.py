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

############################################################################
#
# Pre-processing objects
#
############################################################################

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

    def boundaryLayerLimits( cls , BL_threshold = 0.99 ):

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
        




        

