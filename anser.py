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
lib_path = "./lib"
img_path = "./img"

lib_dir = os.path.join( anser_dir , lib_path )
img_dir = os.path.join( anser_dir , img_path )

sys.path.insert( 0 , lib_dir )
sys.path.insert( 0 , img_dir )


    

