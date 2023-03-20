"""YBW Tow Yo Coordinate Selection"""

import utm
import matplotlib.pyplot as plt

# Globals
REFERENCE = (9.9055599, -104.294361, -2555.)
RX, RY, RN, RL = utm.from_latlon(REFERENCE[0], REFERENCE[1])

# find the corners of the outer box
upper_right = (RX + 300, RY + 300)
upper_left = (RX - 300, RY + 300)
lower_left = (RX - 300, RY - 300)
lower_right = (RX + 300, RY - 300)
lower_right_mid = (RX + 300, RY - 150)
outbox = [lower_right, lower_left, upper_left, upper_right, lower_right_mid]

# find the corners of the inner box
upper_right_small = (RX + 150, RY + 150)
upper_left_small = (RX - 150, RY + 150)
lower_left_small = (RX - 150, RY - 150)
lower_right_small = (RX + 150, RY - 150)
lower_right_end = (RX + 150, RY - 500)
inbox = [lower_left_small, upper_left_small, upper_right_small, lower_right_end]

# convert to lat lons
waypoints = []
for coord in outbox:
    lat, lon = utm.to_latlon(coord[0], coord[1], RN, RL)
    waypoints.append((lat, lon))

for coord in inbox:
    lat, lon = utm.to_latlon(coord[0], coord[1], RN, RL)
    waypoints.append((lat, lon))

# print and plot
print(waypoints)
for coord in waypoints:
    print((coord[0] - 9.) * 60, (coord[1] + 104) * 60.)
# plt.plot([x[1] for x in waypoints], [x[0] for x in waypoints])
# plt.scatter(REFERENCE[1], REFERENCE[0])
# plt.show()

