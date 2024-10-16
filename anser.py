"""
ANSER

Author:     Matthew Holland

This module is the Python software that allows for the reading and post-processing of CFD.

"""

import os
import sys

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

        

