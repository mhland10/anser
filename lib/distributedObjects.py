"""
distributedObjects

Author: Matthew Holland

This library contains all of the objects that Anser uses to read and process CFD data.

"""

import numpy as np
import paraview.simple as pasi
import vtk.util.numpy_support as nps
import vtk
import sys
import pandas as pd

from distributedFunctions import * 

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

    def nutFromKOmega( cls ):
        """
        This method creates a profile for nu_t from k and omega.

        """

        cls.nu_ts = cls.k / cls.omegas

    def nutCap( cls , nut , Chi=None ):
        """
        This method caps the nu_t value at either a given value or Chi value.

        Args:
            nut (float):    [m2/s] nu_t value to be the maximum value.

            Chi (float, optional):  The maxmimum value of Chi to cap nu_t at. Defaults to None,
                                        which caps the value to the input "nut".

        """

        if Chi:
            nut = cls.nu * Chi

        nut_cap = nut * np.ones_like( cls.nu_ts )
        cls.nu_ts = np.maximum( cls.nu_ts , nut_cap )

    def omegaFloor( cls , omega_freestream ):

        cls.omegas[ cls.omegas <= omega_freestream ] = omega_freestream

    def nutFloor( cls , nut_freestream ):

        cls.nu_ts[ cls.nu_ts <= nut_freestream ] = nut_freestream

    def kSmooth( cls , n=10 ):
        """
        Smooths the turbulent kinetic energy profile via a moving average filter.

        Args:
            n (int, optional): The number of terms in the moving average filter. Defaults to 10.

        """

        cls.k = np.convolve( cls.k , np.ones( n ) , mode="same" ) / n

    def nutSmooth( cls , n=10 ):
        """
        Smooths the turbulent viscosity profile via a moving average filter.

        Args:
            n (int, optional): The number of terms in the moving average filter. Defaults to 10.

        """

        cls.nu_ts = np.convolve( cls.nu_ts , np.ones( n ) , mode="same" ) / n

    def kRescale( cls , minK , maxK ):
        """
        Rescale the turbulent kinetic energy according to the inputs.

        Args:
            minK (float):   [m2/s2] The minimum value for the turbulent kinetic energy.

            maxK (float):   [m2/s2] The maximum value for the turbulent kinetic energy.

        """

        C = ( np.max( cls.k ) - np.min( cls.k ) ) / ( maxK - minK )
        cls.k = cls.k / C



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
        data = pasi.OpenDataFile( datafile )
        resample = pasi.ResampleWithDataset()
        resample.SourceDataArrays = [data]
        resample.DestinationMesh = programmableSource
        pasi.UpdatePipeline()

        # Put data in
        self.resampled_output = pasi.servermanager.Fetch(resample)
        point_data = self.resampled_output.GetPointData()
        num_point_arrays = point_data.GetNumberOfArrays()
        self.array_headers = []
        for i in range(num_point_arrays):
            array_name = point_data.GetArrayName(i)
            self.array_headers += [array_name]
        print("Available headers:\t"+str(self.array_headers))

        pasi.Delete( data )
        pasi.Delete( resample )
        pasi.Delete( programmableSource )
        del data
        del resample
        del programmableSource

        # Restore standard output and error to the default
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Optionally suppress VTK messages entirely
        vtk_output_window = vtk.vtkStringOutputWindow()
        vtk.vtkOutputWindow.SetInstance(vtk_output_window)
        vtk.vtkOutputWindow.GetInstance().SetGlobalWarningDisplay(False)

        self.data_loc = "vtk"

        self.coord_change=False
        
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

        #del cls.resampled_output

        cls.data_loc = "dictionary"

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

        cls.data_loc = "pandas"

        #del cls.resampled_output

    def coordinateChange( cls , coord_tol=1e-6 ):
        """
        This method takes the data on the rake and transforms it into the coordinate system defined by
            the rake of normal, tangent, 2nd tangent.

        If one wants a better reference for the methods, one should read:

        Advanced Engineering Mathematics, 6th edition: Chapter 9. By Zill.

        """

        if cls.data_loc.lower()=="pandas":
            
            #
            # Calculate the unit vectors
            #
            cls.C = np.asarray( [ cls.data_df["Cx"].values , cls.data_df["Cy"].values , cls.data_df["Cz"].values ] )

            cls.dC = np.gradient( cls.C , axis=1 )
            cls.tangent_vector = cls.dC / np.linalg.norm( cls.dC , axis=0 )

            cls.dtangent_vector = np.gradient( cls.tangent_vector , axis=1 )
            cls.normal_vector = np.nan_to_num( cls.dtangent_vector / np.linalg.norm( cls.dtangent_vector , axis=0 ) , nan=0 )

            with open("output.txt", "w") as file:
                    # Write the content

                for i , t in enumerate( cls.tangent_vector.T ):

                    cls.normal_vector[:,i]=np.zeros_like(t)

                    #file.write(f"For {i}:\n")
                    #file.write(f"\tTangent vector:\t{str(t)}\n")

                    if np.abs( np.abs(t[0])-1 ) <= coord_tol:

                        #file.write("\tNormal vector to y\n")
                        cls.normal_vector[1,i]=1

                    elif np.abs( np.abs(t[1])-1 ) <= coord_tol:

                        #file.write("\tNormal vector to x\n")
                        cls.normal_vector[0,i]=1

                    elif np.abs( np.abs(t[2])-1 ) <= coord_tol:

                        #file.write("\tNormal vector to x\n")
                        cls.normal_vector[0,i]=1

            cls.binormal_vector = np.cross( cls.tangent_vector , cls.normal_vector , axis=0 )

            #
            # Transform velocity
            #
            cls.U = np.asarray( [ cls.data_df["Ux"].values , cls.data_df["Uy"].values , cls.data_df["Uz"].values ] )

            cls.U_r = np.zeros_like( cls.U )

            for i in range( np.shape( cls.U )[-1] ):

                cls.U_r[1,i] = np.dot( cls.tangent_vector[:,i] , cls.U[:,i] )
                cls.U_r[0,i] = np.dot( cls.normal_vector[:,i] , cls.U[:,i] )
                cls.U_r[2,i] = np.dot( cls.binormal_vector[:,i] , cls.U[:,i] )

            #cls.data_df["C_n"] = cls.normal_vector
            #cls.data_df["C_t"] = cls.tangent_vector
            #cls.data_df["C_b"] = cls.binormal_vector

            cls.data_df["U_n"] = cls.U_r[0,:]
            cls.data_df["U_t"] = cls.U_r[1,:]
            cls.data_df["U_b"] = cls.U_r[2,:]

            cls.C_r = np.zeros_like( cls.C_r )

            for i in range( np.shape( cls.C )[-1] ):

                cls.C_r[0,i] = np.dot( cls.normal_vector[:,i] , cls.C[:,i] )
                cls.C_r[1,i] = np.dot( cls.tangent_vector[:,i] , cls.C[:,i] )
                cls.C_r[2,i] = np.dot( cls.binormal_vector[:,i] , cls.C[:,i] )

            cls.data_df["C_n"] = cls.C_r[0,:]
            cls.data_df["C_t"] = cls.C_r[1,:]
            cls.data_df["C_b"] = cls.C_r[2,:]

        cls.coord_change=True


    def flowData( cls , nu , side=None , dataDictionaryFormat="pandas" , x_offset=0 ):
        """
        This method provides flow data for the rake, assuming proximity to a wall. 


        Args:
            side (string, optional):    The side that the wall is on. The two available options are:

                                        -"LHS": The side of the rake with the lower index values.

                                        -"RHS": The side of the rake with the higher index values.

                                        The default is None, (TODO:NOPE) which automatically detects the wall. Not
                                            case sensitive.

        Raises:
            ValueError: _description_
        """

        if not cls.coord_change:

            if side.lower()=="lhs":
                print("Left hand side flow data.")

                if dataDictionaryFormat.lower()=="pandas":
                    print("Pandas data")

                    cls.u = cls.data_df["U_n"].values
                    cls.y = cls.data_df["C_t"].values
                    #print("Raw y's:\t"+str(cls.y))
                    cls.y[np.abs(cls.y)>0] = cls.y[np.abs(cls.y)>0] * ( cls.y[np.abs(cls.y)>0] / np.abs( cls.y[np.abs(cls.y)>0] ) )
                    #print("Normalized y's:\t"+str(cls.y))
                    cls.x = cls.data_df["C_n"].values[0] - x_offset
                    cls.delta , cls.delta_star , cls.theta = boundaryLayerThickness( cls.y , cls.u )
                    cls.u_tau , cls.C_f = shearConditions( cls.y , cls.u , nu )
                    #print("u_tau:\t{x:.3f}".format(x=cls.u_tau))
                    cls.Re_x = ReynoldsNumber( cls.x , nu , u = cls.u )
                    cls.Re_delta = ReynoldsNumber( cls.delta , nu , u = cls.u )
                    cls.Re_theta = ReynoldsNumber( cls.theta , nu , u = cls.u )
                    cls.Re_tau = ReynoldsNumber( cls.delta , nu , U_inf=cls.u_tau )
                    cls.delta_x = cls.delta / cls.x
                    cls.delta_star_x = cls.delta_star / cls.x
                    cls.theta_x = cls.theta / cls.x
                    cls.H = cls.delta_star / cls.theta
                
            elif side.lower()=="rhs":
                print("Right hand side flow data.")

                if dataDictionaryFormat.lower()=="pandas":
                    print("Pandas data")

                    cls.u = cls.data_df["U_n"].values[::-1]
                    cls.y = cls.data_df["C_t"].values[::-1]
                    #print("Raw y's:\t"+str(cls.y))
                    cls.y[np.abs(cls.y)>0] = cls.y[np.abs(cls.y)>0] * ( cls.y[np.abs(cls.y)>0] / np.abs( cls.y[np.abs(cls.y)>0] ) )
                    #print("Normalized y's:\t"+str(cls.y))
                    cls.x = cls.data_df["C_n"].values[-1] - x_offset
                    cls.delta , cls.delta_star , cls.theta = boundaryLayerThickness( cls.y , cls.u )
                    cls.u_tau , cls.C_f = shearConditions( cls.y , cls.u , nu )
                    #print("u_tau:\t{x:.3f}".format(x=cls.u_tau))
                    cls.Re_x = ReynoldsNumber( cls.x , nu , u = cls.u )
                    cls.Re_delta = ReynoldsNumber( cls.delta , nu , u = cls.u )
                    cls.Re_theta = ReynoldsNumber( cls.theta , nu , u = cls.u )
                    cls.Re_tau = ReynoldsNumber( cls.delta , nu , U_inf=cls.u_tau )
                    cls.delta_x = cls.delta / cls.x
                    cls.delta_star_x = cls.delta_star / cls.x
                    cls.theta_x = cls.theta / cls.x
                    cls.H = cls.delta_star / cls.theta

            elif side==None:
                raise ValueError( "None side not implemented yet" )
            
            else:
                raise ValueError( "Invalid side selected" )
            
        else:

            if side.lower()=="lhs":
                print("Left hand side flow data.")

                if dataDictionaryFormat.lower()=="pandas":
                    print("Pandas data")

                    cls.u = cls.data_df["Ux"].values
                    cls.y = cls.data_df["C_n"].values
                    #print("Raw y's:\t"+str(cls.y))
                    cls.y[np.abs(cls.y)>0] = cls.y[np.abs(cls.y)>0] * ( cls.y[np.abs(cls.y)>0] / np.abs( cls.y[np.abs(cls.y)>0] ) )
                    #print("Normalized y's:\t"+str(cls.y))
                    cls.x = cls.data_df["x"].values[0] - x_offset
                    cls.delta , cls.delta_star , cls.theta = boundaryLayerThickness( cls.y , cls.u )
                    cls.u_tau , cls.C_f = shearConditions( cls.y , cls.u , nu )
                    #print("u_tau:\t{x:.3f}".format(x=cls.u_tau))
                    cls.Re_x = ReynoldsNumber( cls.x , nu , u = cls.u )
                    cls.Re_delta = ReynoldsNumber( cls.delta , nu , u = cls.u )
                    cls.Re_theta = ReynoldsNumber( cls.theta , nu , u = cls.u )
                    cls.Re_tau = ReynoldsNumber( cls.delta , nu , U_inf=cls.u_tau )
                    cls.delta_x = cls.delta / cls.x
                    cls.delta_star_x = cls.delta_star / cls.x
                    cls.theta_x = cls.theta / cls.x
                    cls.H = cls.delta_star / cls.theta
                
            elif side.lower()=="rhs":
                print("Right hand side flow data.")

                if dataDictionaryFormat.lower()=="pandas":
                    print("Pandas data")

                    cls.u = cls.data_df["Ux"].values[::-1]
                    cls.y = cls.data_df["y"].values[::-1]
                    #print("Raw y's:\t"+str(cls.y))
                    cls.y[np.abs(cls.y)>0] = cls.y[np.abs(cls.y)>0] * ( cls.y[np.abs(cls.y)>0] / np.abs( cls.y[np.abs(cls.y)>0] ) )
                    #print("Normalized y's:\t"+str(cls.y))
                    cls.x = cls.data_df["x"].values[-1] - x_offset
                    cls.delta , cls.delta_star , cls.theta = boundaryLayerThickness( cls.y , cls.u )
                    cls.u_tau , cls.C_f = shearConditions( cls.y , cls.u , nu )
                    #print("u_tau:\t{x:.3f}".format(x=cls.u_tau))
                    cls.Re_x = ReynoldsNumber( cls.x , nu , u = cls.u )
                    cls.Re_delta = ReynoldsNumber( cls.delta , nu , u = cls.u )
                    cls.Re_theta = ReynoldsNumber( cls.theta , nu , u = cls.u )
                    cls.Re_tau = ReynoldsNumber( cls.delta , nu , U_inf=cls.u_tau )
                    cls.delta_x = cls.delta / cls.x
                    cls.delta_star_x = cls.delta_star / cls.x
                    cls.theta_x = cls.theta / cls.x
                    cls.H = cls.delta_star / cls.theta

            elif side==None:
                raise ValueError( "None side not implemented yet" )
            
            else:
                raise ValueError( "Invalid side selected" )

    def closeout( cls ):

        del cls.resampled_output
            

class pointDistribution:
    """
    This object allows the user to create point distribution that can be used for the post-
        processing.

    """

    def __init__( self , L , ds , s_0 , normal=(0,1,0) , s_end=None , LHS_endpoint=None , RHS_endpoint=None ):
        """
        Create a point distribution that corresponds to the inputs to initialize the point
            distribution object.

        Args:
            L (float):  [m] The length of the points distribution.

            ds (float): [m] The uniform distance between the points.

            s_0 (float):    [m] The array of the first point coordinates. Must be in format:

                            [ x_0 , y_0 , z_0 ]

            normal (float, optional):   The normal vector for the points to follow.

            s_end (float, optional):    The end point for the 

            LHS_endpoint [float, optional]: The overide coordinates of the first point in
                                                the point distribution.

            RHS_endpoint [float, optional]: The overide coordinates of the last point in
                                                the point distribution.

        Attributes:
            L (float)   <- L

            ds (float)  <- ds

            normal (float)  <-  normal

            x [float]:  The x-coordinates of the point distribution.

            y [float]:  The y-coordinates of the point distribution.

            z [float]:  The z-coordinates of the point distribution.

        """

        #
        # Place the inputs into the object
        #
        self.L = L
        self.ds = ds
        self.N = int( L / ds )
        self.normal = normal / np.linalg.norm( normal )

        #
        # Create uniform point distribution
        #
        if s_end==None:

            x_L = L*self.normal[0]
            x_st = np.minimum( s_0[0] , s_0[0]+x_L )
            x_sp = np.maximum( s_0[0] , s_0[0]+x_L )
            x_step = np.abs( ds*normal[0] )
            y_L = L*self.normal[1]
            y_st = np.minimum( s_0[1] , s_0[1]+y_L )
            y_sp = np.maximum( s_0[1] , s_0[1]+y_L )
            y_step = np.abs( ds*normal[1] )
            z_L = L*self.normal[2]
            z_st = np.minimum( s_0[2] , s_0[2]+z_L )
            z_sp = np.maximum( s_0[2] , s_0[2]+z_L )
            z_step = np.abs( ds*normal[2] )

        elif len( s_end )==3:

            x_L = s_end[0] - s_0[0]
            x_st = s_0[0]
            x_sp = s_end[0]
            y_L = s_end[1] - s_0[1]
            y_st = s_0[1]
            y_sp = s_end[1]
            z_L = s_end[2] - s_0[2]
            z_st = s_0[2]
            z_sp = s_end[2]

            x_step = ds * x_L / np.linalg.norm( [ x_L , y_L , z_L ] )
            y_step = ds * y_L / np.linalg.norm( [ x_L , y_L , z_L ] )
            z_step = ds * z_L / np.linalg.norm( [ x_L , y_L , z_L ] )

            print(f"Length:\t({x_L}, {y_L}, {z_L})")
            print(f"Step\t({x_step}, {y_step}, {z_step})")

            self.normal = [ x_L , y_L , z_L ] / np.linalg.norm( [ x_L , y_L , z_L ] )

        else:

            raise ValueError( "Invalid option. Must be point & normal or point & point" )


        if np.abs(x_L)>0:
            print("Putting in x-values")
            #self.x = np.arange( x_st , x_sp+x_step/10 , np.abs( x_step ) )
            self.x = np.linspace( x_st , x_sp , num=self.N )
            if np.abs(y_L)==0:
                self.y = y_st * np.ones_like( self.x )
            if np.abs(z_L)==0:
                self.z = z_st * np.ones_like( self.x )
        if np.abs(y_L)>0:
            print("Putting in y-values")
            #self.y = np.arange( y_st , y_sp+y_step/10 , np.abs( y_step ) )
            self.y = np.linspace( y_st , y_sp , num=self.N)
            if np.abs(x_L)==0:
                self.x = x_st * np.ones_like( self.y )
            if np.abs(z_L)==0:
                self.z = z_st * np.ones_like( self.y )
        if np.abs(z_L)>0:
            #self.z = np.arange( z_st , z_sp+z_step/10 , np.abs( z_step ) )
            self.z = np.linspace( z_st , z_sp , num=self.N )
            if np.abs(x_L)==0:
                self.x = x_st * np.ones_like( self.z )
            if np.abs(y_L)==0:
                self.y = y_st * np.ones_like( self.z )

        #
        # Add in override points
        #
        if not LHS_endpoint==None:
            self.x[0]=LHS_endpoint[0]
            self.y[0]=LHS_endpoint[1]
            self.z[0]=LHS_endpoint[2]
        if not RHS_endpoint==None:
            self.x[-1]=RHS_endpoint[0]
            self.y[-1]=RHS_endpoint[1]
            self.z[-1]=RHS_endpoint[2]


    def logInsert( cls , c_0 , l , N , side=None ):

        if not hasattr( cls , "x_logs" ):
            cls.x_logs=[]
        if not hasattr( cls , "y_logs" ):
            cls.y_logs=[]
        if not hasattr( cls , "z_logs" ):
            cls.z_logs=[]
        
        #
        # Create the log distributions
        #
        if not side==None and side.lower()=="lhs":

            x_L = np.abs( l*cls.normal[0] )
            y_L = np.abs( l*cls.normal[1] )
            z_L = np.abs( l*cls.normal[2] )

            if x_L>0:
                cls.x_log = cls.x[0] + cls.normal[0] * ( c_0[0] / np.abs( c_0[0] ) ) * np.logspace( np.log10( np.abs( c_0[0] ) ) , np.log10( np.abs( c_0[0] ) + x_L ) , num = N )
            else:
                cls.x_log = cls.x[0] * np.ones(N)
            if y_L>0:
                cls.y_log = cls.y[0] + cls.normal[1] * ( c_0[1] / np.abs( c_0[1] ) ) * np.logspace( np.log10( np.abs( c_0[1] ) ) , np.log10( np.abs( c_0[1] ) + y_L ) , num = N )
            else:
                cls.y_log = cls.y[0] * np.ones(N)
            if z_L>0:
                cls.z_log = cls.z[0] + cls.normal[2] * ( c_0[2] / np.abs( c_0[2] ) ) * np.logspace( np.log10( np.abs( c_0[2] ) ) , np.log10( np.abs( c_0[2] ) + z_L ) , num = N )
            else:
                cls.z_log = cls.z[0] * np.ones(N)

        elif not side==None and side.lower()=="rhs":

            x_L = np.abs( l*cls.normal[0] )
            y_L = np.abs( l*cls.normal[1] )
            z_L = np.abs( l*cls.normal[2] )
        
            if x_L>0:
                cls.x_log = cls.x[-1] - cls.normal[0] * ( c_0[0] / np.abs( c_0[0] ) ) * np.logspace( np.log10( np.abs( c_0[0] ) ) , np.log10( np.abs( c_0[0] ) + x_L ) , num = N )
            else:
                cls.x_log = cls.x[-1] * np.ones(N)
            if y_L>0:
                cls.y_log = cls.y[-1] - cls.normal[1] * ( c_0[1] / np.abs( c_0[1] ) ) * np.logspace( np.log10( np.abs( c_0[1] ) ) , np.log10( np.abs( c_0[1] ) + y_L ) , num = N )
            else:
                cls.y_log = cls.y[-1] * np.ones(N)
            if z_L>0:
                cls.z_log = cls.z[-1] - cls.normal[2] * ( c_0[2] / np.abs( c_0[2] ) ) * np.logspace( np.log10( np.abs( c_0[2] ) ) , np.log10( np.abs( c_0[2] ) + z_L ) , num = N )
            else:
                cls.z_log = cls.z[-1] * np.ones(N)

        else:

            x_L = l*cls.normal[0]
            x_st = np.minimum( c_0[0] , c_0[0]+x_L )
            x_sp = np.maximum( c_0[0] , c_0[0]+x_L )
            if np.abs(x_L)>0:
                cls.x_log = cls.normal[0] * np.logspace( np.log10(np.abs(x_st)) , np.log10(np.abs(x_sp)) , num=N)
            else:
                cls.x_log = x_st * np.ones( N )
            y_L = l*cls.normal[1]
            y_st = np.minimum( c_0[1] , c_0[1]+y_L )
            y_sp = np.maximum( c_0[1] , c_0[1]+y_L )
            if np.abs(y_L)>0:
                cls.y_log = cls.normal[1] * np.logspace( np.log10(np.abs(y_st)) , np.log10(np.abs(y_sp)) , num=N)
            else:
                cls.y_log = y_st * np.ones( N )
            z_L = l*cls.normal[2]
            z_st = np.minimum( c_0[2] , c_0[2]+z_L )
            z_sp = np.maximum( c_0[2] , c_0[2]+z_L )
            if np.abs(z_L)>0:
                cls.z_log = cls.normal[2] * np.logspace( np.log10(np.abs(z_st)) , np.log10(np.abs(z_sp)) , num=N)
            else:
                cls.z_log = z_st * np.ones( N )

        #
        # Insert the log distributions
        #
        if not side==None and side.lower()=="lhs":

            cls.x_r = np.append( cls.x_log , cls.x[1:] )
            cls.y_r = np.append( cls.y_log , cls.y[1:] )
            cls.z_r = np.append( cls.z_log , cls.z[1:] )

        elif not side==None and side.lower()=="rhs":

            cls.x_r = np.append( cls.x[:-1] , cls.x_log[::-1] )
            cls.y_r = np.append( cls.y[:-1] , cls.y_log[::-1] )
            cls.z_r = np.append( cls.z[:-1] , cls.z_log[::-1] )

        else:

            if np.abs( cls.normal[0] )==np.max( np.abs( cls.normal ) ):
                if np.min(cls.x)<np.min(cls.x_log):
                    cls.x_r = cls.x[cls.x<np.min(cls.x_log)]
                    cls.y_r = cls.y[cls.x<np.min(cls.x_log)]
                    cls.z_r = cls.z[cls.x<np.min(cls.x_log)]
                    cls.x_r = np.append( cls.x_r , cls.x_log )
                    cls.y_r = np.append( cls.y_r , cls.y_log )
                    cls.z_r = np.append( cls.z_r , cls.z_log )
                else:
                    cls.x_r = cls.x_log
                    cls.y_r = cls.y_log
                    cls.z_r = cls.z_log
                if np.max(cls.x)>np.max(cls.x_log):
                    cls.x_r = np.append( cls.x_r , cls.x[cls.x>np.max(cls.x_log)] )
                    cls.y_r = np.append( cls.y_r , cls.y[cls.x>np.max(cls.x_log)] )
                    cls.z_r = np.append( cls.z_r , cls.z[cls.x>np.max(cls.x_log)] )
                
            elif np.abs( cls.normal[1] )==np.max( np.abs( cls.normal ) ):
                if np.min(cls.y)<np.min(cls.y_log):
                    cls.x_r = cls.x[cls.y<np.min(cls.y_log)]
                    cls.y_r = cls.y[cls.y<np.min(cls.y_log)]
                    cls.z_r = cls.z[cls.y<np.min(cls.y_log)]
                    cls.x_r = np.append( cls.x_r , cls.x_log )
                    cls.y_r = np.append( cls.y_r , cls.y_log )
                    cls.z_r = np.append( cls.z_r , cls.z_log )
                else:
                    cls.x_r = cls.x_log
                    cls.y_r = cls.y_log
                    cls.z_r = cls.z_log
                if np.max(cls.y)>np.max(cls.y_log):
                    cls.x_r = np.append( cls.x_r , cls.x[cls.y>np.max(cls.y_log)] )
                    cls.y_r = np.append( cls.y_r , cls.y[cls.y>np.max(cls.y_log)] )
                    cls.z_r = np.append( cls.z_r , cls.z[cls.y>np.max(cls.y_log)] )

            else:
                if np.min(cls.z)<np.min(cls.z_log):
                    cls.x_r = cls.x[cls.z<np.min(cls.z_log)]
                    cls.y_r = cls.y[cls.z<np.min(cls.z_log)]
                    cls.z_r = cls.z[cls.z<np.min(cls.z_log)]
                    cls.x_r = np.append( cls.x_r , cls.x_log )
                    cls.y_r = np.append( cls.y_r , cls.y_log )
                    cls.z_r = np.append( cls.z_r , cls.z_log )
                else:
                    cls.x_r = cls.x_log
                    cls.y_r = cls.y_log
                    cls.z_r = cls.z_log
                if np.max(cls.z)>np.max(cls.z_log):
                    cls.x_r = np.append( cls.x_r , cls.x[cls.z>np.max(cls.z_log)] )
                    cls.y_r = np.append( cls.y_r , cls.y[cls.z>np.max(cls.z_log)] )
                    cls.z_r = np.append( cls.z_r , cls.z[cls.z>np.max(cls.z_log)] )
        
        cls.x = cls.x_r
        cls.y = cls.y_r
        cls.z = cls.z_r

    def initBL( cls , nu , u_tau , dyplus_0 , side , stream=(1,0,0) ):
        """
        Initialize a Boundary Layer for the object to create a distribution of points.

        Args:
            nu (float): [m2/s] The kinematic viscosity of the fluid.

            u_tau (float):  [m/s] The friction velocity of the boundary layer.

            dyplus_0 (float):   [-] The resolution of the first step.

            side (string):  Which side of the point distribution the BL will be defined on.
                                The valid options are:

                            -"lhs": The smaller index, or Left Hand Side.

                            -"rhs": The larger index, or Right Hand Side.

                            Not case sensitive.

            stream (float, optional):   The vector that points along the streamwise direction of the
                                            boundary layer.

        Attributes:
            BL {}:  The boundary layer dictionary that contains all the data needed to produce
                        the log distribution. The entries will be:

                    -"nu":  [m2/s] The kinematic viscosity of the fluid.

                    -"u_tau":   [m/s] The friction velocity of the boundary layer.

                    -"dyplus_0":    [-] The y+ height of the first step.

                    -"dy_0":    [m] The y height of the first step.

                    -"stream_vector":   [m] The vector of the streamwise direction for the boundary 
                                                layer.

        """

        if not hasattr( cls , "BL" ):
            cls.BL=[]

        BL_init={}
        BL_init["nu"]=nu
        BL_init["u_tau"]=u_tau
        BL_init["dyplus_0"]=dyplus_0
        BL_init["dy_0"]=dyplus_0*nu/u_tau
        if side.lower()=="lhs" or side.lower()=="rhs":
            BL_init["side"]=side.lower()
        else:
            raise ValueError( "side must be either LHS or RHS" )
        BL_init["stream_vector"]=np.asarray( stream ) / np.linalg.norm( np.asarray( stream ) )

        cls.BL+=[BL_init]

    def distBL( cls , selection=None , fit_tolerance = 1e-3 ):

        if not selection==None:
            cls.off_normal = cls.normal - np.dot( cls.normal , cls.BL[selection]["stream_vector"] ) * cls.BL[selection]["stream_vector"] / ( np.linalg.norm( cls.BL[selection]["stream_vector"] ) ** 2 )

            if np.abs( cls.off_normal[1] ) > 0:
                y_BL_height = 0.1
                y_BL_N = 25
                if cls.BL[selection]["side"]=="lhs":
                    y_BL = np.logspace( np.min( cls.y ) , np.min( cls.y ) + y_BL_height , num=y_BL_N )
                    y_BL_init_error = y_BL[0] / y_BL_N
                else:
                    y_BL = np.logspace( np.max( cls.y ) - y_BL_height , np.max( cls.y ) , num=y_BL_N )
                

        else:
            print("Hello there")


