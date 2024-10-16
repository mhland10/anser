"""
distributedObjects

Author: Matthew Holland

This library contains all of the objects that Anser uses to read and process CFD data.

"""

import numpy as np

###################################################################################################
#
# Data Objects
#
###################################################################################################

class boundaryLayer:

    def __init__( self , vonKarmanConst = 0.41 , vanDriestConst = 5.0 , distDomainLims = [ 1e-3 , 1e3 ] , distDomainN = 1e3 ):
        """
        This object is the data object that contains the data and methods to define the boundary
            layer.

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

        vonKarmanConst : <float>    The von Karman constant that will be used to define the law-of-
                                        the-wall distribution.

        vanDriestConst : <float>    The constant that define the van Driest profile of the law-of-
                                        the-wall distribution.

        ypluss : [float]    The y+ domain to find the law-of-the-wall over.

        Upluss : [float]    The values of U+ that define the law-of-the wall distribution.
 
        """
    
        self.vonKarmanConst = vonKarmanConst
        self.vanDriestConst = vanDriestConst
    
        self.ypluss = np.logspace( np.log10( np.min( distDomainLims ) ) , np.log10( np.max( distDomainLims ) ) , num = int( distDomainN ) )
        self.Upluss = np.zeros( int( distDomainN ) )
        for i , ypl in enumerate( self.ypluss ):
            if ypl > 10 :
                self.Upluss[i] = ( 1 / self.vonKarmanConst ) * np.log( ypl ) + self.vanDriestConst
            else:
                self.Upluss[i] = ypl


###################################################################################################
#
# 
#
###################################################################################################