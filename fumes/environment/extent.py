
class Extent(object):
    """Axis-aligned geographic extent object."""

    def __init__(self, xrange, xres, yrange, yres,
                 zrange=None, zres=None, global_origin=(0., 0., 0.)):
        """Initialize geographic extent.

        Args:
            xrange (float): x range of the world, in meters
            xres (int): number of discrete bins used in the x direction
            yrange (float): y range of the world, in meters
            yres (int): number of discrete bins used in the y direction
            zrange (float): z range of the world, in meters altitude
            zres (int): number of discrete bins used in the z direction
            global_origin (tuple[float]): (lat, lon, depth) coordinate of
                the world origin.
        """
        self.origin = global_origin

        self.xrange = xrange
        self.xmin = xrange[0]
        self.xmax = xrange[1]
        self.xres = xres

        self.yrange = yrange
        self.ymin = yrange[0]
        self.ymax = yrange[1]
        self.yres = yres

        if zrange is not None:
            self.zrange = zrange
            self.zmin = zrange[0]
            self.zmax = zrange[1]
            self.zres = zres
        else:
            self.zrange = [0, 0]
            self.zmin = 0
            self.zmax = 0
            self.zres = 1

    def get_attributes(self):
        """Returns dict with Extent definition."""
        return {'xrange': self.xrange,
                'xres': self.xres,
                'yrange': self.yrange,
                'yres': self.yres,
                'zrange': self.zrange,
                'zres': self.zres,
                'global_origin': self.origin}
