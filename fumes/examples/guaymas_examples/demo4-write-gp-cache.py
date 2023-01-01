"""Demonstrates GP cache reading and writing crossflow environment."""

from frontmatter_crossflow import *

# Experiment name
experiment_name = "crossflow_gp_cache"

tic()
print("Getting snapshots.")
tvec = np.linspace(0, 12 * 3600, 3)
for i, t in enumerate(tvec):
    print(f"Getting snapshot {t}: {i} of {len(tvec)}")
    mtt.get_snapshot(t=t,
                     xrange=[-500, 500], xres=20,
                     yrange=[-500, 500], yres=20,
                     z=np.linspace(0, 100, 5),
                     from_cache=False)
print("Done.")
toc()

###################
# Test Writing the GP cache
###################
tic()
print("Writing GP cache.")
mtt.write_gp_cache(tvec=tvec,
                   xrange=[-500, 500], xres=20,
                   yrange=[-500, 500], yres=20,
                   zrange=[0, 100], zres=5,
                   overwrite=True,
                   visualize=True)
print("Done.")
toc()

tic()
print("Reading GP cache.")
gp_model = mtt.read_gp_cache()
print("Done.")
toc()

print("Getting cached value.")

query_point = (np.linspace(10, 10, 1),
               np.linspace(10, 10, 1),
               np.linspace(200, 200, 1))

tic()
print("Value from model: ", mtt.get_value(t=75,
                                          loc=query_point,
                                          from_cache=False))
toc()
tic()
print("Value from cache: ", mtt.get_value(t=75,
                                          loc=query_point,
                                          from_cache=True,
                                          cache_interp="gp"))
toc()
tic()
print("Again, value from cache: ", mtt.get_value(t=75,
                                                 loc=query_point,
                                                 from_cache=True,
                                                 cache_interp="gp"))
toc()
