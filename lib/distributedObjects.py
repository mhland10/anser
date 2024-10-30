"""
distributedObjects

Author: Matthew Holland

This library contains all of the objects that Anser uses to read and process CFD data.

"""

import numpy as np
import paraview.simple as pasi
import vtk.util.numpy_support as nps
import pandas as pd

###################################################################################################
#
# Data Objects
#
###################################################################################################

class boundaryLayer:

    def __init__( self , vonKarmanConst = 0.41 , vanDriestConst = 5.0 , distDomainLims = [ 1e-3 , 1e3 ] , distDomainN = 1e3 , regionSwitch = 11 ):
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

        **regionSwitch : <float>    The y+ value where the behavior switches from the inner to outer 
                                        region.

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
        self.regionSwitch = regionSwitch
    
        self.ypluss = np.logspace( np.log10( np.min( distDomainLims ) ) , np.log10( np.max( distDomainLims ) ) , num = int( distDomainN ) )
        self.Upluss = np.zeros( int( distDomainN ) )
        for i , ypl in enumerate( self.ypluss ):
            if ypl > regionSwitch :
                self.Upluss[i] = ( 1 / self.vonKarmanConst ) * np.log( ypl ) + self.vanDriestConst
            else:
                self.Upluss[i] = ypl

    def colesProfile( cls , C = 4.7 , eta_1 = 11 , b = 0.33 ):
        """
        This method changes the profile to Coles' profile described in his 1956 paper.

        Args:

            C (float, optional):    Constant C from distribution. Defaults to 4.7.

            eta_1 (float, optional):    Eta constant from distribution. Defaults to 11.

            b (float, optional):    b constant from distribution. Defaults to 0.33.

        """

        C_1 = - np.log( cls.vonKarmanConst ) / cls.vonKarmanConst + C

        cls.Upluss = np.log( 1 + cls.vonKarmanConst * cls.ypluss ) / cls.vonKarmanConst
        cls.Upluss = cls.Upluss + C_1 * ( 1 - np.exp( - cls.ypluss / eta_1 ) - ( cls.ypluss / eta_1 ) * np.exp( - cls.ypluss * b ) )

    def wakeProfile( cls , plus_delta_conversion , Pi = 0.25 ):
        """
        Adds in a wake profile for the velocity profile distribution.

        Args:
            plus_delta_conversion (float):  [1/m] The multiplier to go from the wall units to
                                                boundary layer units.

            Pi (float, optional):   The Pi constant from the distribution. Defaults to 0.25.
        
        """
        
        # y/delta values
        cls.y_delta = cls.ypluss * plus_delta_conversion

        # Add wake to profile
        cls.Upluss = cls.Upluss + ( 1 / cls.vonKarmanConst ) * ( ( cls.y_delta ** 2 ) - ( cls.y_delta ** 3 ) + 6 * Pi * ( cls.y_delta ** 2 ) - 4 * Pi * ( cls.y_delta ** 3 ) )
    
    def nu_tProfile( cls , nu , nu_t_visc=0.0 , E=9.8 , blending_function="exponential" , n=4.0 ):
        """
        This method adds turbulent kinematic viscosity to the boundary layer profile based on
            the velocity profile.

        The source for this is the OpenFOAM User Guide at:

        https://www.openfoam.com/documentation/guides/latest/doc/guide-bcs-wall-turbulence-nutkWallFunction.html
        https://www.openfoam.com/documentation/guides/latest/doc/guide-bcs-wall-turbulence-nutWallFunction.html

        Args:
            nu (float): [m2/s] The kinematic viscosity of the fluid.

            nu_t_visc (float, optional):    The turbulent kinematic viscosity of the viscous
                                                subregion. Defaults to 0.0.
                                                
            E (float, optional):    Roughness parameter. Defaults to 9.8.
            
            blending_function (string, optional):   The blending function that will transition
                                                        from viscous to log regions. The valid
                                                        options are:

                                                    -"discontinuous":   Uses the maximum value
                                                                            of viscosity.

                                                    -"binomial":    Combines via an exponential
                                                                        norm, of power "n".

                                                    -*"exponential":    Combines via an 
                                                                            exonential function.

                                                    The default is "exponential". Not case
                                                        sensitive.

            n (float, optional):    The exponent for the binomial blending function.

        """

        cls.nu = nu

        cls.nu_t_viscs = nu_t_visc * np.zeros_like( cls.Upluss )
        cls.nu_t_logs = nu * ( cls.ypluss * cls.vonKarmanConst / np.log( E * cls.ypluss ) - 1 )

        if blending_function.lower()=="exponential":
            cls.Gammas = 0.01 * ( cls.ypluss ** 4 ) / ( 1.0 + 5.0 * cls.ypluss )
            cls.nu_ts = cls.nu_t_viscs * np.exp( - cls.Gammas ) + cls.nu_t_logs * np.exp( - 1 / cls.Gammas )
        
        elif blending_function.lower()=="binomial":
            cls.nu_ts = ( ( cls.nu_t_viscs ** n ) + ( cls.nu_t_logs ** n ) ) ** ( 1 / n )

        elif blending_function.lower()=="discontinuous":
            cls.nu_ts = np.maximum( cls.nu_t_viscs , cls.nu_t_logs )

        else:
            raise ValueError( "Invalid option for blending function." )
        
    def nu_tildaProfile( cls , N_scan=int(1e3) , c_v1 = 7.1 ):
        """
        Creates a profile for nu_tilda from the profile of nu_t according to the Spalart-Allmaras
            model.

        For reference:
            https://en.wikipedia.org/wiki/Spalart%E2%80%93Allmaras_turbulence_model

        Note: One must run "nu_tProfile()" first.

        Args:
            N_scan (int, optional): The number of sample points to scan along. Defaults to int(1e3).
            
            c_v1 (float, optional): The blending function constant for nu_t and nu_tilda. Defaults 
                                        to 7.1.

        Raises:
            ValueError: _description_
        """

        cls.nu_tildas = np.zeros_like( cls.nu_ts )
        for i in range( len( cls.nu_tildas ) ):
            scan = np.logspace( np.log10( cls.nu_ts[i] ) - 3 , np.log10( cls.nu_ts[i] ) + 3 , num = N_scan )
            chis = scan / cls.nu 
            f_v1s = ( chis ** 3 ) / ( ( chis ** 3 ) + ( c_v1 ** 3 ) )
            convergences = ( scan * f_v1s ) / cls.nu_ts[i] - 1
            cls.nu_tildas[i] = np.interp( 0 , convergences , scan )
        
        cls.nu_tildas = np.nan_to_num( cls.nu_tildas , nan=0.0 )

    def kProfile( cls , u_tau , C=11.0 , C_k=-0.416 , B_k=8.366 , C_eps2=1.9 , floor_value=1e-12 ):
        """
        This method produces a profile for turbulent kinetic energy in accordance with the 
            kLowReWallFunction in OpenFOAM.

        For reference:
            https://www.openfoam.com/documentation/guides/latest/doc/guide-bcs-wall-turbulence-kLowReWallFunction.html

        Args:
            u_tau (float):  [m/s] The friction velocity of the boundary layer.

            C (float, optional):    Model coefficient. Defaults to 11.0.

            C_k (float, optional):  Model coefficient. Defaults to -0.416.

            B_k (float, optional):  Model coefficient. Defaults to 8.366.

            C_eps2 (float, optional):   Model coefficient. Defaults to 1.9.

            floor_value (float, optional):  Minimum value in case the value of k becomes too low for
                                                floating point operation in the model. Defaults to 
                                                1e-9.

        """

        cls.C_fs = ( ( cls.ypluss + C ) ** -2 ) + 2 * cls.ypluss / ( C ** 3) - ( C ** -2 )

        cls.k_log = C_k * np.log( cls.ypluss ) / cls.vonKarmanConst + B_k
        cls.k_vis = 2.4e3 * cls.C_fs / C_eps2

        cls.k = np.zeros_like( cls.ypluss )
        for i , yp in enumerate( cls.ypluss ):
            if yp > cls.regionSwitch:
                cls.k[i] = np.max( [ cls.k_log[i] * ( u_tau ** 2 ) , floor_value ] )
            else:
                cls.k[i] = np.max( [ cls.k_vis[i] * ( u_tau ** 2 ) , floor_value ] )

        cls.k_floor = floor_value

    def omegaProfile( cls , nu , u_tau , beta_1=0.075 , C_mu = 0.09 , blending_function="exponential" , n=2.0 ):
        """
        Defines the profile of omega in the boundary layer according to the omegaWallFunction in 
            OpenFOAM.

        For reference:
            https://www.openfoam.com/documentation/guides/latest/doc/guide-bcs-wall-turbulence-omegaWallFunction.html

        Args:
            nu (float):     [m2/s] The kinematic viscosity of the fluid

            u_tau (float):  [m/s] The friction velocity of the boundary layer.

            beta_1 (float, optional):   Coefficient. Defaults to 0.075.

            C_mu (float, optional):     Coefficient. Defaults to 0.09.

            blending_function (str, optional):  Blending function to produce the omega profile. The
                                                    valid options are:
                                                    
                                                -"stepwise":    Switches between the viscous and 
                                                                    log region via wall 
                                                                    coordinate.

                                                -"maximum":     The maximum value of omega will 
                                                                    define the distribution.

                                                -"binomial":    Omega distribution will be the 
                                                                    norm of power "n" between 
                                                                    the viscous and log omega 
                                                                    values.

                                                -*"exponential":    The exponential blending
                                                                        function.
                                                
                                                -"tanh":    The hyperbolic tangent function is used
                                                                to blend the omega profile.

                                                Defaults to "exponential".

            n (float, optional):    The power to take the norm to for binomial blending function. 
                                        Defaults to 2.0.

        """

        cls.ys = cls.ypluss * nu / u_tau

        cls.omega_viss = 6 * nu / ( beta_1 * ( cls.ys ** 2 ) )
        cls.omega_logs = np.sqrt( cls.k ) / ( C_mu * cls.vonKarmanConst * cls.ys )

        if blending_function.lower()=="stepwise":
            cls.omegas = np.zeros_like( cls.ypluss )
            cls.omegas[cls.ypluss<=cls.regionSwitch] = cls.omega_viss[cls.ypluss<=cls.regionSwitch]
            cls.omegas[cls.ypluss>cls.regionSwitch] = cls.omega_logs[cls.ypluss>cls.regionSwitch]

        elif blending_function.lower()=="maxmimum":
            cls.omegas = np.maximum( cls.omega_viss , cls.omega_logs )

        elif blending_function.lower()=="binomial":
            cls.omegas = ( ( cls.omega_viss ** n ) + ( cls.omega_logs ** n ) ) ** (1/n)

        elif blending_function.lower()=="exponential":
            cls.Gammas = 0.01 * ( cls.ypluss ** 4 ) / ( 1.0 + 5.0 * cls.ypluss )

            cls.omegas = cls.omega_viss * np.exp( - cls.Gammas ) + cls.omega_logs * np.exp( - 1 / cls.Gammas )

        elif blending_function.lower()=="tanh":
            cls.phis = np.tanh( ( cls.ypluss / 10 ) ** 4 )

            cls.omega_1s = cls.omega_viss + cls.omega_logs
            cls.omega_2s = ( ( cls.omega_viss ** 1.2 ) + ( cls.omega_logs ** 1.2 ) ) ** (1/1.2)

            cls.omegas = cls.phis * cls.omega_1s + ( 1 - cls.phis ) * cls.omega_2s

    def kFromOmega( cls , limit_method = "floor" ):
        """
        This method creates a profile for k from omega and nu_t.

        """

        cls.k = cls.nu_ts * cls.omegas

        if limit_method.lower()=="floor":
            cls.k = np.maximum( cls.k , np.zeros_like( cls.k ) )
        elif limit_method.lower()=="none":
            cls.k = cls.k

    def omegaFromK( cls ):
        """
        This method creates a profile for omega from k and nu_t.

        """

        cls.omegas = cls.k / cls.nu_ts



###################################################################################################
#
# Post-Processing Objects
#
###################################################################################################

class rake:
    """
    This object is a rake of points that allows the user to draw data from the datafiles to draw
        the wanted data.
    
    """

    def __init__( self , points , datafile ):
        """
        Initialize the rake object according to the inputs to the file.

        The data will be stored in a Paraview-native format.

        Args:

            points ((arrays/lists)):    The tuple of arrays or lists that contain the points of
                                            the rakes. The order will be (x, y, z). All three
                                            dimensions are required.

            datafile (string):  The datafile with the CFD data.

        Attributes:

            ext_points [list]:  The externally defined points from "points" re-formatted into a
                                    Paraview-friendly format.

            
        
        """

        # Check the number of dimensions
        if not len( points ) == 3:
            raise ValueError( "Not enough dimensions in points. Make sure three (3) dimensions are present.")
        
        # Load the data from the *.vtk files
        data = pasi.OpenDataFile( datafile )

        # Change the format of the points
        self.ext_points = []
        for i in range( len( points[0] ) ):
            self.ext_points += [[ points[0][i] , points[1][i] , points[2][i] ]]

        # Create the rake in Paraview
        programmableSource = pasi.ProgrammableSource()
        programmableSource.OutputDataSetType = 'vtkPolyData'
        programmableSource.Script = f"""
        import vtk

        # Manually input the external points
        custom_points = {self.ext_points}

        # Create a vtkPoints object to store the points
        points = vtk.vtkPoints()

        # Insert custom points into the vtkPoints object
        for point in custom_points:
            points.InsertNextPoint(point)

        # Create a polyline (a single line connecting the points)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(custom_points))
        for i in range(len(custom_points)):
            lines.InsertCellPoint(i)

        # Create the output PolyData and assign points and lines
        output.SetPoints(points)
        output.SetLines(lines)
        """

        # Pull data from rake
        self.data = pasi.OpenDataFile( datafile )
        self.resample = pasi.ResampleWithDataset()
        self.resample.SourceDataArrays = [data]
        self.resample.DestinationMesh = programmableSource
        pasi.UpdatePipeline()

        # Put data in
        self.resampled_output = pasi.servermanager.Fetch(self.resample)
        self.point_data = self.resampled_output.GetPointData()
        self.num_point_arrays = self.point_data.GetNumberOfArrays()
        self.array_headers = []
        for i in range(self.num_point_arrays):
            array_name = self.point_data.GetArrayName(i)
            self.array_headers += [array_name]

    def dataToDictionary( cls ):
        """
        Transfers the data from the Paraview-native format to a Python-native format of a
            dictionary.

        Attributes:

            data {}:    The dictionary containing the data from the rake.

        """

        cls.data = {}

        #
        # Get the coordinates
        #
        points_vtk = cls.resampled_output.GetPoints().GetData()
        points_np = np.asarray( nps.vtk_to_numpy( points_vtk ) )
        for i , c in enumerate( [ 'x' , 'y' , 'z' ] ):
            cls.data[c] = points_np[:,i]

        #
        # Get the data
        #
        for i , d in enumerate( cls.array_headers ):
            data_vtk = cls.resampled_output.GetPointData().GetArray( d )
            data_np = nps.vtk_to_numpy( data_vtk )
            cls.data[d] = data_np

    def dataToPandas( cls , coords = ['x', 'y', 'z'] ):
        """
        Put the data from the Paraview native format to Pandas. Pandas will be more convenient for
            exporting.

        Args:
            coords (list, optional):    The coordinates for the data. Defaults to ['x', 'y', 'z'].
        """
        

        #
        # Initialize the dataframe with the coordinates
        #
        points_vtk = cls.resampled_output.GetPoints().GetData()
        points_np = nps.vtk_to_numpy( points_vtk )
        cls.data_df = pd.DataFrame( points_np , columns = coords )

        #
        # Put data in dataframe
        #
        for i , d in enumerate( cls.array_headers ):
            data_vtk = cls.resampled_output.GetPointData().GetArray( d )
            data_np = nps.vtk_to_numpy( data_vtk )
            if len( data_np.shape ) > 1 :
                for j , c in enumerate( coords ):
                    data_name = d + c
                    cls.data_df[data_name] = data_np[:,j]
            else:
                cls.data_df[d] = data_np


