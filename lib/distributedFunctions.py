"""
Distrbuted Functions of ANSER

This file contains the functions that ANSER uses to provide CFD post-processing.

Author(s):  Matthew Holland

Changelog

Version     Date        Description

0.0         2024/10/15  Original Version of the Distributed Functions

"""

################################################################################################
#
# Import required modules
#
################################################################################################

import numpy as np
import os
import paraview.simple as pasi

################################################################################################
#
# David's Functions
#
################################################################################################

"""
These functions come from the work that David Paeres has performed. No modifications have been
    done.

"""

def fnc_generateStartEndPoints_forNew_CurvilinearMesh(min_x, max_x, new_nx, Length, y):
    ''' 
    fnc_generateStartEndPoints_forNew_CurvilinearMesh:
        Inputs:
        * min_x:    Scalar of the X-location where the probe lines are going to start
        * max_x:    Scalar of the X-location where the probe lines are going to end
        * new_nx:   Scalar of how many stations (uniformly distributed) are going to be sampled or probed.
        * Length:   Scalar of the probe lines' length (sampling lines arc length), starting from the wall surface.
        * y:        Function defined containing the piecewise math formulation that describes wall surface geometry.
        Returns:
        * points_x: 2D Array of shape [new_nx,2] containing the X-coordinates' start&end-points of all probe lines.
        * points_y: 2D Array of shape [new_nx,2] containing the Y-coordinates' start&end-points of all probe lines.
        * tangent_x: 1D Array of the flow direction angles with respect to the x-axis for every station probed.
    '''
    
    ### Allocating Start&End Points and Tangent Angle Arrays
    points_x  = np.zeros((new_nx,2))
    points_y  = np.zeros_like(points_x)
    tangent_x = np.zeros_like(points_x[:,0])

    ### X-coordinate values for the probe lines' StartPoints
    points_x[:,0]  = np.linspace(min_x,max_x,new_nx)

    ### Y-coordinate values for the probe lines' StartPoints
    for i in range(new_nx):
        points_y[i,0]  = y(points_x[i,0])                       # Evaluating X-coordinate values' StartPoints in the function y(x)
        tangent_x[i] = derivative(y, points_x[i,0], dx=1e-6)    # Tangent angles (θ) from the numerical derivatives (dy/dx)
    
    cosTheta = np.cos(tangent_x)                                # True dx = cos(θ)
    sinTheta = np.sin(tangent_x)                                # True dy = sin(θ)
    
    ### X & Y coordinates values for the probe lines' EndPoints
    points_x[:,1] = points_x[:,0] + (-sinTheta*Length)          # EndPointX = StartPointX + -sin(θ)*Length
    points_y[:,1] = points_y[:,0] + (cosTheta*Length)           # EndPointY = StartPointY + cos(θ)*Length

    return points_x, points_y, tangent_x 

def fnc_plotOverLine_exportAs_csv(VTK, newNx, newNy, points_x, points_y, Nz):
    ''' 
    fnc_plotOverLine_exportAs_csv: 
        Inputs:
        * VTK:      String of the VTK filename with the full path.
        * newNx:    Scalar of how many stations (uniformly distributed) are going to be sampled or probed.
        * newNy:    Scalar of how many wall-levels (uniformly distributed) are going to be probed (sampling lines arc resolution).
        * points_x: 2D Array of shape [new_nx,2] containing the X-coordinates' start&end-points of all probe lines.
        * points_y: 2D Array of shape [new_nx,2] containing the Y-coordinates' start&end-points of all probe lines.
        * Nz:       Scalar of how many crossflow slices the computational domain has.
        Returns:    
        * CSVfilesPath: String of the general name (using wildcard *) with the path of all .csv files containing all interpolated 
                        data of the probe lines limited by 25 probe lines per .csv file.
    P.S. This function was built from adapting an script previously made with Paraview's Python Trace
    '''
    ### Making sure this is a 2D case
    if Nz==1:
        points_z = 0
    else :
        print("Is this a 2D case? Why Nz is not equal 1?")

    ### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    ### Reading the VTK file using 'Legacy VTK Reader'
    VTKdata = LegacyVTKReader(registrationName= 'VTKdata', FileNames=[f'{VTK}'])

    # create a new 'Gradient' of Pressure
    UpdatePipeline(time=0.0, proxy=VTKdata)
    gradient1 = Gradient(registrationName='Gradient1', Input=VTKdata)
    gradient1.ScalarArray = ['POINTS', PressureScalarField_name]                             
    gradient1.ResultArrayName = 'Gradient_p'                            # Here Pressure was assumed to be called as 'p'

    # create a new 'Gradient' of Velocity
    UpdatePipeline(time=0.0, proxy=gradient1)
    gradient2 = Gradient(registrationName='Gradient2', Input=gradient1)
    gradient2.ScalarArray = ['POINTS', VelocityVectorField_name]                             
    gradient2.ResultArrayName = 'Gradient_U'                            # Here Velocity was assumed to be named as 'U'
    
    ### Fetch the data object associated with the dataset
    UpdatePipeline(time=0.0, proxy=gradient2)
    data_object = servermanager.Fetch(gradient2)

    ### Check if data_object has point data
    if data_object.GetPointData():
        PointData_names = []
        point_data = data_object.GetPointData()
        for i in range(point_data.GetNumberOfArrays()):
            PointData_names.append(point_data.GetArrayName(i))

    ### Due the heavy computational memory required in high-resolution probe lines, the data is saved as csv file every 25 lines
    count = 0 
    group = 0
    ProbeLinesData=0
    for i in range(0,newNx): 
        if count == 5:

            CSVPath = Path(str(VTK).replace(".vtk",f"_{group}.csv"))
            SaveData(f'{CSVPath}', proxy=ProbeLinesData,PointDataArrays= PointData_names,Precision=9,UseScientificNotation=1)

            ProbeLinesData=0
            group += 1
            count = 0

        # create a new 'Plot Over Line'
        plotOverLine = PlotOverLine(registrationName='PlotOverLine', Input= gradient2)
    
        # init the 'Line' selected for 'Source'
        plotOverLine.Point1 = [points_x[i,0], points_y[i,0], points_z]
        plotOverLine.Point2 = [points_x[i,1], points_y[i,1], points_z]
        plotOverLine.Resolution = newNy-1 

        UpdatePipeline(time=0.0, proxy=plotOverLine)

        ProbeLinesData = AppendDatasets(registrationName='ProbeLinesData', Input=[ProbeLinesData, plotOverLine])
        ProbeLinesData.OutputDataSetType = 'Unstructured Grid'

        count += 1

    # save data
    CSVPath = Path(str(VTK).replace(".vtk",f"_{group}.csv"))
    SaveData(f'{CSVPath}', proxy=ProbeLinesData,PointDataArrays= PointData_names,Precision=9,UseScientificNotation=1,)

    # general name with path of all csv files
    CSVfilesPath = str(VTK).replace(".vtk","_*.csv")

    return CSVfilesPath , VTKdata

def solve_nan(y):

        def nan_helper(y):
            """Helper to handle indices and logical indices of NaNs.
            Input:
                - y, 1d numpy array with possible NaNs
            Output:
                - nans, logical indices of NaNs
                - index, a function, with signature indices= index(logical_indices),
                to convert logical indices of NaNs to 'equivalent' indices
            Example:
                >>> # linear interpolation of NaNs
                >>> nans, x= nan_helper(y)
                >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
            """
            return np.isnan(y), lambda z: z.nonzero()[0]
    
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        return y

def rotateVectorComponents_in2D(comp_x, comp_y, tangent_x):
    # Perform the rotation to obtain the curvilinear velocity components (Us, Un)
    # Us =  Ux * cos(θ) + Uy * sin(θ)
    # Un = -Ux * sin(θ) + Uy * cos(θ)
    # OR, Transform (dP/dx),(dP/dy) to (dP/ds),(dP/dn)
    # dP/ds = dP/dx * cos(θ) + dP/dy * sin(θ)
    # dP/dn = -dP/dx * sin(θ) + dP/dy * cos(θ)
    nx = len(tangent_x)
    cosTheta = np.cos(tangent_x).reshape((nx,1))                # True dx = cos(θ)
    sinTheta = np.sin(tangent_x).reshape((nx,1))                # True dy = sin(θ)
    comp_s = np.array(comp_x).reshape((nx,-1)) * cosTheta  +  np.array(comp_y).reshape((nx,-1)) * sinTheta
    comp_n = np.array(comp_y).reshape((nx,-1)) * cosTheta  + -np.array(comp_x).reshape((nx,-1)) * sinTheta

    return comp_s.reshape(-1), comp_n.reshape(-1)

###################################################################################################
#
# Flow Data Functions
#
###################################################################################################

def boundaryLayerThickness( y , u , U_inf=None , threshold = 0.99 , y_min = 1e-6 , n_samples = 100 ):
    """
    Calculate the boundary layer thicknesses for a given flow profile.

    Args:
        y [float]:  [m] The wall-normal coordinates for the flow profile.

        u [float]:  [m/s] The streamwise velocity of the flow profile.

        U_inf (float, optional):    The freestream velocity of the flow profile. Defaults to None, 
                                        which makes the freestream velocity the maximum velocity of
                                        "u".

        threshold (float, optional):    The threshold that defines the edge of the boundary layer by
                                            the ratio of velocity to the freestream velocity 
                                            (u/U_inf). Defaults to 0.99.

        y_min (float, optional):    [m] The minimum wall-normal coordinate for the projected flow 
                                        profile. Note that in the function, there is a new domain
                                        created between the wall and boundary layer edge. Defaults 
                                        to 1e-6.

        n_samples (int, optional):  The number of samples in the projected flow profile. Defaults 
                                        to 100.

    Returns:
        
        BLthickness (float):    [m] The boundary layer thickness according to the flow profile and
                                    prescribed threshold.

        disp_BLthickness (float):   [m] The displacement boundary layer thickness.

        mom_BLthickness (float):    [m] The momentum boundary layer thickness.

    """

    # Define the freestream velocity if not already defined
    if not U_inf:
        U_inf = np.max( u )
    
    # Calculate velocity ratio
    u_U = u / U_inf
    print("u/U:\t"+str(u_U))
    u_U_limited = u_U[:np.where(u_U>=1)[0][0]]
    y_limited = y[:np.where(u_U>=1)[0][0]]
    print("u/U (limited):\t"+str(u_U_limited))

    # Find the boundary layer thickness
    BLthickness = np.interp( threshold , u_U_limited , y_limited )

    # Project onto a new domain between the wall and the edge of the boundary layer
    y_projected = np.logspace( np.log10( y_min ) , np.log10( BLthickness ) , num = n_samples )
    u_U_projected = np.interp( y_projected , y_limited , u_U_limited )

    # Calculate the displacement and momentum boundary layer thicknesses from integral definitions
    disp_BLthickness = np.trapz( ( 1 - u_U_projected ) , x = y_projected )
    mom_BLthickness = np.trapz( u_U_projected * ( 1 - u_U_projected )  , x = y_projected )

    return BLthickness , disp_BLthickness , mom_BLthickness

def shearConditions( y , u , nu , U_inf=None ):
    """
    Calculates the shear conditions of a boundary layer given the flow profile.

    Args:
        y [float]:      [m] Wall-normal coordinates.

        u [float]:      [m/s] Streamwise velocity profile.

        nu (float):     [m2/s] Kinematic viscosity.

        UU_inf (float, optional):    The freestream velocity of the flow profile. Defaults to None, 
                                        which makes the freestream velocity the maximum velocity of
                                        "u".

    Returns:
        u_tau (float):  [m/s] Friciton velocity.

        C_f (float):    [-] Skin friction coefficient.
        
    """


    # Define the freestream velocity if not already defined
    if not U_inf:
        U_inf = np.max( u )

    # Calculate shear velocity
    u_tau = np.sqrt( nu * np.abs( ( u[1] - u[0] ) / ( y[1] - y[0] ) ) )
    
    # Skin friction coefficient
    C_f = 2 * ( ( u_tau / U_inf ) ** 2 )

    return u_tau , C_f 

def ReynoldsNumber( x , nu , u=None , U_inf=None ):
    """
    This function creates a Reynolds number from the required inputs.

    Args:
        x (float):      [m] The distance measurement of Reynolds number. Can be 0 for unit length.

        nu (float):     [m2/s] The kinematic viscosity of the flow.

        u [float, optional]:    [m/s] The velocity profile of the flow. Defaults to None, must be
                                    numeric if "U_inf" is None.

        U_inf (float, optional): [m/s] The freestream velocity of the flow. Defaults to None.

    Returns:
        Re (float):     [-] The Reynolds number according to the input parameters.

    """

    if not U_inf:
        U_inf = np.max( u )
        
    if x==0:
        x = 1

    Re = U_inf * x / nu
        
    return Re
