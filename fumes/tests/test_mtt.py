import numpy as np
import matplotlib.pyplot as plt

from fumes.environment.utils import pacific_sp_S, pacific_sp_T, eos_rho
from fumes.environment import CrossflowMTT
from fumes.model import MTT

def test_mtt():
	pass
	# z = np.linspace(0, 200, 100)
	# mtt = BentMTT(z,
	# 				tprof=pacific_sp_T,
	# 				sprof=pacific_sp_S,
	# 				rhoprof=eos_rho,
	# 				curfunc=lambda x, t: 0.2,
	# 				headfunc=lambda t: 45.0 * np.pi / 180.0,
	# 				vex=0.4,
	# 				area=0.1,
	# 				salt=34.608,
	# 				temp=300,
	# 				density=eos_rho(300, 34.608))
	# output = ['Velocity', 'Area', 'Salinity', 'Temperature', 'Crossflow Dist']
	# compare = [None, None, pacific_sp_S, pacific_sp_T, None]

	# height = 100
	# x = np.linspace(-500, 500, 1000)
	# y = np.linspace(-500, 500, 1000)
	# pv = mtt.get_value(3.0, (x, y, height))
	# ps = mtt.get_snapshot(None)
