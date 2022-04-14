"""Allows arbitrary meters-based coordinates to be converted to lat/lon.

To be used in conjunction with any script that produces navigational
coordinates in meters (e.g., in the fumes package, the output missions).
Provides a commandline interface, or can be imported and called as
a tool independently.
"""
import numpy as np
import utm


def convert_to_latlon(coords, latlon_origin):
    """Computes lat-lon coordinates from arbitrary meters and origin."""
    # get reference frame of the new origin
    easting, northing, zone_number, zone_letter = utm.from_latlon(
        latlon_origin[0], latlon_origin[1])
    # add easting, northing to appropriate coords
    map_ncoords = np.asarray([nc + northing for nc in coords[:, 1]])
    map_ecoords = np.asarray([ec + easting for ec in coords[:, 0]])
    # now convert back to latlon coordinates
    map_lat, map_lon = utm.to_latlon(
        map_ecoords, map_ncoords, zone_number, zone_letter)

    return np.hstack([map_lon.reshape(-1, 1),
                      map_lat.reshape(-1, 1),
                      (coords[:, 2] + latlon_origin[2]).reshape(-1, 1)])
