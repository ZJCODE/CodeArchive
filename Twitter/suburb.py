# File: suburb.py
# Author: Jack Massey
# Date: 2015-04-13
# Licence: None
#
# Used to map point to "suburbs" or geometric shapes imported from a shapefule.
# Usage:
#        suburbs = suburbs_get(<filename>)
#        s = suburbs_find(suburbs, <point>)
#        print(suburbs["info"][s]["record"])
################################################################################

import shapefile
from shapely import geometry
from rtree import index


# suburbs_get
# Load a shapefile and create a suburb dictionary
# Reuturns a dictionary with the rtree, a dictionary of the shapefile and a
# dictionary of the population (will be all 0 for now).
# Arguments: source - filename for source shapefile
# Returns: A suburb dictionary
def suburbs_get(source = "../data/SSC06aAUST/SSC06aAUST_region"):
    sf = shapefile.Reader(source)
    shapeRecords = sf.shapeRecords()
    idx = index.Index()
    info = {}
    data = {}
    for shapeRecord in shapeRecords:
        try:
            idx.insert(int(shapeRecord.record[0]), shapeRecord.shape.bbox)
            info[str(shapeRecord.record[0])] = {
                "shape": geometry.asShape(shapeRecord.shape),
                "record": shapeRecord.record
            }
            data[str(shapeRecord.record[0])] = 0
        except AttributeError:
            print "Reading Error: " + shapeRecord.record[0] + " " + shapeRecord.record[1] + " " + shapeRecord.record[2]
    return {"rtree": idx, "info": info, "pop": data}

# suburb_find
# Tests and finds the suburb that a given point is contained in.
# Arguments: point - a tupple with the latitude and longitude
# Returns: The name of the suburb that the point is in.
def suburbs_find(suburbs, point):
    #idx, data, count = suburbs

    # For every suburb based on bounding box
    for i in list(suburbs["rtree"].intersection(point)):
        check = geometry.Point(point)
        if suburbs["info"][str(i)]["shape"].contains(check):
            return str(i)
    raise LookupError
