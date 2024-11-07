"""
ANSER

Author:     Matthew Holland

This module is the Python software that allows for the reading and post-processing of CFD.

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# Boundary Layer Profile Objects
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

    def colesProfile( cls ):

        cls.profile.colesProfile()

    def wake( cls , delta , nu , u_tau , Pi=0.25 , Pi_search=False , Pi_range = ( 0.1 , 1 ) , Pi_N = 10 , theta=None , theta_only=True ):
        """
        Create a wake for the boundary layer according to the input parameters.

        Args:
            delta (float):  [m] Boundary layer height.

            nu (float):     [m2/s] The kinematic viscosity of the fluid.

            u_tau (float):  [m/s] The friction velocity of the boundary layer.

            Pi (float):     The value of Pi that will be used to generate the wake profile.

            Pi_search (boolean, optional):  If Pi will be found, the switch to turn that on.

            Pi_range [float, optional]: The range of values to search for Pi across.

            Pi_N (float, optional): The number of samples to search for Pi over.
        """
        
        cls.delta_plus_conversion = nu / ( u_tau * delta )

        if Pi_search:

            if not theta and not hasattr(cls , 'theta' ):
                raise ValueError("Pi search requires a known value for momentum thickness")

            cls.Pis = np.logspace( np.log10( np.min( Pi_range ) ) , np.log10( np.max( Pi_range ) ) , num = Pi_N )
            cls.deltas = np.zeros_like( cls.Pis )
            cls.thetas = np.zeros_like( cls.Pis )
            for i, Pi in enumerate( cls.Pis ):
                cls.profile.wakeProfile( cls.delta_plus_conversion , Pi = Pi )
                U_pluss = cls.profile.Upluss
                y_pluss = cls.profile.ypluss
                U_pluss[ U_pluss >= ( cls.U_inf / u_tau ) ] =  cls.U_inf / u_tau
                delta_plus , _ , theta_plus = boundaryLayerThickness( y_pluss , U_pluss , y_min=np.min(y_pluss)/10 )
                cls.deltas[i] = delta_plus * ( nu / u_tau )
                cls.thetas[i] = theta_plus * ( nu / u_tau )

            if theta_only:
                cls.convs = -1 + np.sqrt( ( ( cls.thetas / theta ) ** 2 ) )
            else:
                cls.convs = -1 + np.sqrt( ( ( cls.thetas / theta ) ** 2 ) + ( ( cls.deltas / delta ) ** 2 ) )

            if ( np.min( cls.convs ) < 0 ) and ( np.max( cls.convs ) > 0 ):
                Pi = np.interp( 0 , cls.convs , cls.Pis )
            else:
                cls.search = np.gradient( cls.convs )
                Pi = np.interp( 0 , cls.search , cls.Pis ) 


        cls.profile.wakeProfile( cls.delta_plus_conversion , Pi = Pi )

        cls.Pi = Pi
        cls.nu = nu
        cls.u_tau = u_tau

        cls.profile.Upluss[ cls.profile.Upluss >= ( cls.U_inf / u_tau ) ] =  cls.U_inf / u_tau 

    def turbulenceProfile( cls , model = "all"  ):
        """
        Creates the profile of turbulence for the boundary layer being created.

        Args:
            model (str, optional):  The turbulence model to create the profile for. The valid
                                        options are:
                                        
                                    -"sa" or "spalartallmaras" or "spalart-allmaras":
                                        Create the turbulence profile that accounts for the 
                                            parameters. 

                                    -"ko" or "kosst" or "komega:    Use the k-omega or k-omega 
                                                                        sst model.

                                    -*"all":    Create all the known turbulence parameters that
                                                    are available.
                                            
                                    Defaults to "all".

        """

        cls.profile.nu_tProfile( cls.nu )

        if model.lower()=="all" or model.lower()=="sa" or model.lower()=="spalartallmaras" or model.lower()=="spalart-allmaras":
            cls.profile.nu_tildaProfile()
        
        if model.lower()=="all" or model.lower()[:1]=="ko" or model.lower()=="kosst":
            cls.profile.kProfile( cls.u_tau )
            cls.profile.omegaProfile( cls.nu , cls.u_tau )
        

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
        """
        Find the best location to take the recycled BL from via a bisecting search method.

        Note as of 2024/11/07:  This method doesn't work.

        """

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
            cls.y_vals = -np.logspace( np.log10( cls.start_height ) , np.log10( cls.rake_length - cls.start_height ) , num = cls.N_samples )
            cls.x_vals_LHS = boundLHS * np.ones_like( cls.y_vals )
            cls.x_vals_RHS = boundRHS * np.ones_like( cls.y_vals )
            cls.z_vals = np.zeros_like( cls.y_vals )
            cls.rakeLHS = rake( ( cls.x_vals_LHS , cls.y_vals  , cls.z_vals ) , cls.datafile )
            cls.rakeRHS = rake( ( cls.x_vals_RHS , cls.y_vals  , cls.z_vals ) , cls.datafile )
            cls.rakeLHS.dataToDictionary()
            cls.rakeRHS.dataToDictionary()
            print("Rake LHS dictionary keys:\t"+str(cls.rakeLHS.data.keys()))

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
                searching = True
            else:
                searching = False


            #searching = False


        cls.x_bestfit=np.mean([boundLHS,boundRHS])

    def interpolationSearch( cls , N_points, store_data=True ):
        """
        Find the best location for the recycled BL from an interpolation of minimum error.

        Args:
            N_points (int): The number of samples for the interpolation.

            store_data (bool, optional):    Whether or not to store the error data that goes into 
                                                the interpolation. Defaults to True.

        """


        cls.x_vals = np.linspace( np.min( cls.bounding_points ) , np.max( cls.bounding_points ) , num = N_points)
        cls.y_vals = np.append( [0] , -np.logspace( np.log10( cls.start_height ) , np.log10( cls.rake_length - cls.start_height ) , num = cls.N_samples ) )
        cls.z_vals = np.zeros_like( cls.y_vals )
        
        cls.layers_data=[]
        cls.layers_errors=[]
        cls.net_errors=[]

        for i in range( N_points ):
            
            print("i:\t{x}".format(x=i))

            cls.point_coords = ( cls.x_vals[i]*np.ones_like( cls.y_vals ) , cls.y_vals  , cls.z_vals )
            cls.rake0 = rake( cls.point_coords  , cls.datafile )
            cls.rake0.dataToDictionary()

            #
            # Calculate flow parameters
            #
            cls.layer_data = [0]*3
            # BL thicknesses
            cls.layer_data[0] , cls.layer_data[1] , cls.layer_data[2] = boundaryLayerThickness( np.abs( cls.rake0.data['y'] ) , cls.rake0.data['U'][:,0] )
            # Shape factor
            cls.layer_data += [ cls.layer_data[1] / cls.layer_data[2] ]
            # Shear parameters
            cls.layer_data += [0]*2
            cls.layer_data[-2] , cls.layer_data[-1] = shearConditions( np.abs( cls.rake0.data['y'] ) , cls.rake0.data['U'][:,0] , cls.nu )
            # Reynolds numbers
            cls.layer_data += [0]*3
            cls.layer_data[-3] = ReynoldsNumber( 0 , cls.nu , u = cls.rake0.data['U'][:,0] )
            cls.layer_data[-2] = ReynoldsNumber( cls.layer_data[2] , cls.nu , u = cls.rake0.data['U'][:,0] )
            cls.layer_data[-1] = ReynoldsNumber( cls.layer_data[0] , cls.nu , U_inf = cls.layer_data[3] )

            #
            # Calculate normalized error
            #
            cls.layer_errors = np.asarray( cls.layer_data ) / cls.boundaryLayer_data - 1
            cls.layer_errorNorm = np.linalg.norm( cls.layer_errors )
            print("\tLayer norm:\t{x:.3f}".format(x=cls.layer_errorNorm))

            if store_data:
                cls.layer_data+=[cls.layer_data]
                cls.layers_errors+=[cls.layer_errors]
            cls.net_errors+=[cls.layer_errorNorm]

        cls.x_bestfit = np.interp( 0 , cls.net_errors , cls.x_vals  )

    def recycledBLPull( cls , target="default" , turbulence_headers=None , separated=False , p_value=None ):

        y_points = np.append( [0] , -np.logspace( np.log10( cls.start_height ) , np.log10( cls.rake_length - cls.start_height ) , num = cls.N_samples ) )
        x_points = cls.x_bestfit * np.ones_like( y_points )
        z_points = np.zeros_like( y_points )
        cls.rake_BL = rake( ( x_points , y_points , z_points ) , cls.datafile )
        
        if target.lower()=="default":

            cls.rake_BL.dataToPandas()
            cls.df_export=cls.rake_BL.data_df

            if p_value:
                cls.df_export["p"]=p_value

            cls.columns_to_export=[ "y" , "Ux" , "Uy" , "Uz"  ]

            if not separated:
                cls.columns_to_export+=[ "p" ]

                if not turbulence_headers==None:
                    cls.columns_to_export+=turbulence_headers

                cls.df_export[cls.columns_to_export].to_csv("recycled_profile.csv",index=False)

            else:

                cls.df_export[cls.columns_to_export].to_csv("recycled_velocity_profile.csv",index=False)

                if turbulence_headers:
                    for i,t in enumerate( turbulence_headers ):
                        cls.df_export[["y",t]].to_csv("recycled_"+t+"_profile.csv",index=False)
                
                cls.df_export[["y","p"]].to_csv("recycled_"+t+"_profile.csv",index=False)

        if target.lower()=="openfoam":

            cls.rake_BL.dataToDictionary()
            cls.data_export=cls.rake_BL.data

            #
            # Export full
            #
            if separated:

                cls.sorted_u_data = sorted( zip( cls.data_export["y"], cls.data_export["U"] ), key=lambda x: x[0] )

                formatted_data=["("]
                formatted_data += [ f"({cls.sorted_u_data[i][0]} ({cls.sorted_u_data[i][1][0]} {cls.sorted_u_data[i][1][1]} {cls.sorted_u_data[i][1][2]}))" for i in range( len( cls.sorted_u_data ) )]
                formatted_data+=[")"]

                with open("recycled_velocity_profile.dat",'w') as f:
                    for line in formatted_data:
                        f.write( line + "\n" )

                for j , t in enumerate( turbulence_headers ):
                    cls.sorted_t_data = sorted( zip( cls.data_export["y"] , cls.data_export[t] ) , key=lambda x: x[0] )

                    formatted_data=["("]
                    formatted_data += [ f"({cls.sorted_t_data[i][0]} {cls.sorted_t_data[i][1]})" for i in range( len( cls.sorted_t_data ) )]
                    formatted_data+=[")"]

                    with open("recycled_"+t+"_profile.dat",'w') as f:
                        for line in formatted_data:
                            f.write( line + "\n" )

                #cls.sorted_p_data = sorted( zip( cls.data_export["y"] , cls.data_export["p"] ) , key=lambda x: x[0] )

                print("Hello there")

###################################################################################################
#
# Lead-In Profile
#
###################################################################################################

class leadInBL:

    def __init__( self , theta_target , delta_target , L_domain , U_inf , nu ):
        """
        Initialize the lead-in Boundary Layer object.

        Calculates the inlet conditions from the target conditions from the following relations
            from Kays and Crawford:

            Re_theta = 0.036 * ( Re_x ^ 0.8 )

        And the following from White and Majdalani:

            Re_delta = 0.016 * ( Re_x ^ (6/7) )

        Args:
            theta_target (float):   [m] The momentum thickness to target.

            L_domain (float):   [m] The length of the domain to develop to the target boundary 
                                    layer.

            U_inf (float):  [m/s] The effective freestream velocity of the boundary layer.

            nu (float): [m2/s] Kinematic viscosity of the flow.

        """

        #
        # Store the parameters
        #
        self.theta_1 = theta_target
        self.delta_1 = delta_target
        self.L = L_domain
        self.U_inf = U_inf
        self.nu = nu

        #
        # Find inlet conditions
        #
        self.theta_0 = ( ( self.U_inf * self.theta_1 / self.nu ) - 0.036 * ( ( self.U_inf * self.L / self.nu ) ** 0.8 ) ) * ( self.nu / self.U_inf )
        self.delta_0 = ( ( self.U_inf * self.delta_1 / self.nu ) - 0.016 * ( ( self.U_inf * self.L / self.nu ) ** (6/7) ) ) * ( self.nu / self.U_inf )

    def profileGenerate( cls , u_tau , turbulence_model="all" ):
        """
        Generate the profile of the inlet boundary layer

        Args:
            u_tau (float):  [m/s] Shear velocity

            turbulence_model (str, optional):  The turbulence model to create the profile for. The 
                                        valid options are:
                                        
                                    -"sa" or "spalartallmaras" or "spalart-allmaras":
                                        Create the turbulence profile that accounts for the 
                                            parameters. 

                                    -"ko" or "kosst" or "komega:    Use the k-omega or k-omega 
                                                                        sst model.

                                    -*"all":    Create all the known turbulence parameters that
                                                    are available.
                                            
                                    Defaults to "all".

        """

        cls.BL = syntheticBoundaryLayer( distDomainLims=[ 1e-1 , cls.delta_0 * u_tau / cls.nu ] , distDomainN=100 )
        cls.BL.dimensionalize( cls.U_inf , cls.nu , u_tau=u_tau )
        cls.BL.profile.colesProfile()
        cls.BL.wake( cls.delta_0 , cls.nu , u_tau )

        cls.BL.turbulenceProfile( model=turbulence_model )

        cls.u_tau = u_tau
        cls.turb_model = turbulence_model

    def velocityProfileExport( cls , h_domain , normal=(0,1,0) , N_freestream = 100 , target = "default" ):
        
        cls.y = normal[1] * cls.BL.profile.ypluss * cls.nu / cls.u_tau
        cls.y = np.append( np.zeros(1) , cls.y )
        cls.y = np.append( cls.y , np.linspace( normal[1] * np.max( np.abs( cls.y ) ) * 1.1 , h_domain , num = N_freestream ) )
        cls.u = cls.BL.profile.Upluss * cls.u_tau
        cls.u = np.append( np.zeros(1) , cls.u )
        cls.u = np.append( cls.u , cls.U_inf * np.ones( N_freestream ) )
        cls.v = np.zeros_like( cls.u )
        cls.w = np.zeros_like( cls.u )

        opening = ['('] * len( cls.y )
        closing = [')'] * len( cls.y )

        if target.lower()=="default":
            
            df = pd.DataFrame( { 'y' : cls.y , 'U_x': cls.u , 'U_y' : cls.v , 'U_z' : cls.w } )
            df.to_csv( 'velocity_profile.csv' , index=False )

        elif target.lower()=="openfoam":

            sorted_indices = np.argsort(cls.y)

            # Sort all arrays using the sorted indices
            y_sorted = cls.y[sorted_indices]
            u_sorted = cls.u[sorted_indices]
            v_sorted = cls.v[sorted_indices]
            w_sorted = cls.w[sorted_indices]
            
            # Format data to include parentheses
            formatted_data = ["("]
            formatted_data += [f"({y_sorted[i]}\t({u_sorted[i]} {v_sorted[i]} {w_sorted[i]}))" for i in range(len(y_sorted))]
            formatted_data += [")"]

            # Write to a .dat file
            with open('velocity_profile.dat', 'w') as f:
                for line in formatted_data:
                    f.write(line + "\n")

            cls.sort_indx = sorted_indices

        cls.N_free = N_freestream
        cls.target = target

    def turbulenceProfileExport( cls ):


        #
        # Write turbulent kinematic viscosity
        #
        cls.nu_t = np.append( np.zeros(1) , cls.BL.profile.nu_ts )
        cls.nu_t = np.append( cls.nu_t , cls.nu_t[-1] * np.ones( cls.N_free ) )

        if cls.target.lower()=="default":
            
            df = pd.DataFrame( { 'y' : cls.y , 'nut': cls.nu_t } )
            df.to_csv( 'nut_profile.csv' , index=False )

        elif cls.target.lower()=="openfoam":

            # Sort all arrays using the sorted indices
            y_sorted = cls.y[cls.sort_indx]
            nut_sorted = cls.nu_t[cls.sort_indx]
            
            # Format data to include parentheses
            formatted_data = ["("]
            formatted_data += [f"({y_sorted[i]}\t{nut_sorted[i]})" for i in range(len(y_sorted))]
            formatted_data += [")"]

            # Write to a .dat file
            with open('nut_profile.dat', 'w') as f:
                for line in formatted_data:
                    f.write(line + "\n")

        #
        # Write SA model parameters
        #
        if cls.turb_model.lower()=="all" or cls.turb_model.lower()=="sa" or cls.turb_model.lower()=="spalartallmaras" or cls.turb_model.lower()=="spalart-allmaras":
            cls.nu_tilda = np.append( np.zeros(1) , cls.BL.profile.nu_tildas )
            cls.nu_tilda = np.append( cls.nu_tilda , cls.nu_tilda[-1] * np.ones( cls.N_free ) )

            if cls.target.lower()=="default":
            
                df = pd.DataFrame( { 'y' : cls.y , 'nuTilda': cls.nu_tilda } )
                df.to_csv( 'nuTilda_profile.csv' , index=False )

            elif cls.target.lower()=="openfoam":

                # Sort all arrays using the sorted indices
                y_sorted = cls.y[cls.sort_indx]
                nut_sorted = cls.nu_tilda[cls.sort_indx]

                # Format data to include parentheses
                formatted_data = ["("]
                formatted_data += [f"({y_sorted[i]}\t{nut_sorted[i]})" for i in range(len(y_sorted))]
                formatted_data += [")"]

                # Write to a .dat file
                with open('nuTilda_profile.dat', 'w') as f:
                    for line in formatted_data:
                        f.write(line + "\n")
        
        #
        # Write k-o model parameters
        #
        if cls.turb_model.lower()=="all" or cls.turb_model.lower()[:1]=="ko" or cls.turb_model.lower()=="kosst":
            cls.k = np.append( np.zeros(1) , cls.BL.profile.k )
            cls.k = np.append( cls.k , cls.k[-1] * np.ones( cls.N_free ) )

            cls.omega = np.append( cls.BL.profile.omegas[0] , cls.BL.profile.omegas )
            cls.omega = np.append( cls.omega , cls.omega[-1] * np.ones( cls.N_free ) )

            if cls.target.lower()=="default":
            
                df = pd.DataFrame( { 'y' : cls.y , 'k': cls.k } )
                df.to_csv( 'k_profile.csv' , index=False )

            elif cls.target.lower()=="openfoam":

                # Sort all arrays using the sorted indices
                y_sorted = cls.y[cls.sort_indx]
                k_sorted = cls.k[cls.sort_indx]

                # Format data to include parentheses
                formatted_data = ["("]
                formatted_data += [f"({y_sorted[i]}\t{k_sorted[i]})" for i in range(len(y_sorted))]
                formatted_data += [")"]

                # Write to a .dat file
                with open('k_profile.dat', 'w') as f:
                    for line in formatted_data:
                        f.write(line + "\n")

            if cls.target.lower()=="default":
            
                df = pd.DataFrame( { 'y' : cls.y , 'omega': cls.omega } )
                df.to_csv( 'omega_profile.csv' , index=False )

            elif cls.target.lower()=="openfoam":

                # Sort all arrays using the sorted indices
                y_sorted = cls.y[cls.sort_indx]
                omega_sorted = cls.omega[cls.sort_indx]

                # Format data to include parentheses
                formatted_data = ["("]
                formatted_data += [f"({y_sorted[i]}\t{omega_sorted[i]})" for i in range(len(y_sorted))]
                formatted_data += [")"]

                # Write to a .dat file
                with open('omega_profile.dat', 'w') as f:
                    for line in formatted_data:
                        f.write(line + "\n")

###################################################################################################
#
# OpenFOAM Post-Processing
#
###################################################################################################

class caseReader:
    """
    This object allows a user to read an OpenFOAM case and pull data efficiently given a proper set
        up.

    """

    def __init__( self , casename , casepath , working_directory=None ):
        """
        Initialize the case reader for an OpenFOAM case.

        Args:
            casename (string):  The name of the case.

            casepath (string):  The path of the case in the system to move to the directory.

            working_directory (string, optional):   The directory to store data in. The defaul is
                                                        None, leave as if "casepath" is also the
                                                        working directory.

        """


        self.casename = casename
        self.casepath = casepath

        if working_directory:
            self.working_directory = working_directory
        else:
            self.working_directory = self.casepath

    def convergencePlot( cls , headers , residualfile=None , residualpath=None , residualimg="residuals.png" , img_directory=None , preprocess=False ):
        """
        This method plots the residuals from the post-processing data files.

        Args:
            headers [string]:   The headers of the dataframe to use.

            residualfile (string, optional):    The name of the file where the residuals are 
                                                    stored. Defaults to None, leave as is to 
                                                    automatically detect.

            residualpath (string, optional):    The path to find the residuals. Defaults to 
                                                    None, leave as is to automatically detect.

            residualimg (str, optional):    The name of the image to store the residual plot
                                                into. The default is "residuals.png".

            img_directory (str, optional):  The name of the directory in the working directory
                                                to save the files to. The default is None,
                                                leave as is to store in the working directory.

        """

        os.chdir( cls.casepath )
        os.chdir( "./postProcessing/residuals/" )

        # List only directories in the current directory
        dirs = [d for d in os.listdir() if os.path.isdir(d)]

        # Check if there is only one directory
        if len(dirs) == 1:
            os.chdir(dirs[0])
            print(f"Changed directory to: {dirs[0]}")
        else:
            print("Error: There is not exactly one directory present.")
            os.chdir(dirs[-1])

        # Read the table, skipping the comment lines
        if not residualfile:
            residualfile = "residuals.dat"    

        if preprocess:
            # Read all lines from the file
            with open(residualfile, "r") as file:
                lines = file.readlines()

            # Modify the second line by removing its first character
            lines[1] = lines[1][1:]

            # Delete the third line
            lines.pop(2)

            # Write the modified content back to the file
            with open(residualfile, "w") as file:
                file.writelines(lines)
        if headers:
            cls.df_residuals = pd.read_csv( residualfile , delimiter="\t", comment='#' , na_values=0 )
        else:
            cls.df_residuals = pd.read_csv( residualfile , delimiter="\t", comment='#', header=0)
        
        cls.df_residuals.columns = cls.df_residuals.columns.str.strip()

        # Plot the data
        os.chdir( cls.working_directory )
        if img_directory:
            os.chdir( img_directory )
        else:
            os.chdir( cls.working_directory )
        for c in cls.df_residuals.columns[1:]:
            print("Plot "+c)
            plt.semilogy( cls.df_residuals['Time'] , cls.df_residuals[c] , label=c )
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        plt.title('Residuals Over Time')
        plt.legend(loc='best')
        plt.savefig( residualimg , dpi=300 , bbox_inches='tight' )
        plt.show()
        os.chdir( cls.working_directory )

        





        

