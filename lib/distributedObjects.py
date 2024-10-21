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
    
        self.ypluss = np.logspace( np.log10( np.min( distDomainLims ) ) , np.log10( np.max( distDomainLims ) ) , num = int( distDomainN ) )
        self.Upluss = np.zeros( int( distDomainN ) )
        for i , ypl in enumerate( self.ypluss ):
            if ypl > regionSwitch :
                self.Upluss[i] = ( 1 / self.vonKarmanConst ) * np.log( ypl ) + self.vanDriestConst
            else:
                self.Upluss[i] = ypl
    


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


