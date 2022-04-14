"""Saving utilities"""

import json
import scipy as sp
import numpy as np
from scipy import interpolate
from distfit import distfit
from fumes.model.parameter import Parameter
from fumes.environment.current import CurrMag, CurrHead
from fumes.environment.extent import Extent
from fumes.environment.profile import Profile
from fumes.environment.utils import eos_rho


def initialize_param_from_json(dict):
    """Returns parameter initialized from json dict."""
    if dict['dist_name'] is not 'is_scalar':
        prior = distfit(distr=dict['dist_name'])
        dist = getattr(sp.stats, dict['dist_name'])
        if dict['dist_arg']:
            X = dist.rvs(*dict['dist_arg'], loc=dict['dist_loc'],
                         scale=dict['dist_scale'], size=10000)
        else:
            X = dist.rvs(loc=dict['dist_loc'], scale=dict['dist_scale'], size=10000)
        prior.fit_transform(X)
    else:
        prior = dict['dict_loc']

    if dict['prop_name'] is not 'is_scalar':
        proposal = sp.stats.norm(loc=dict['prop_loc'], scale=dict['prop_scale'])
    else:
        proposal = dict['prop_loc']

    return Parameter(prior, proposal)


def initialize_extent_from_json(dict):
    """Returns extent initialized from json dict."""
    return Extent(**dict)


def initialize_model_from_json(json_file):
    """Returns all of the intialization needed."""
    f = open(json_file)
    data = json.load(f)

    stat_dict = {}

    fixed = data['model_fixed_params']
    stat_dict['plume_loc'] = tuple(fixed['plume_loc'])
    stat_dict['extent'] = initialize_extent_from_json(fixed['extent']),
    stat_dict['temp'] = fixed['temp']
    stat_dict['salt'] = fixed['salt']
    stat_dict['density'] = fixed['density']
    z = fixed['z']
    stat_dict['tprof'] = Profile(**fixed['tprof'])
    stat_dict['sprof'] = Profile(**fixed['sprof'])
    stat_dict['rhoprof'] = eos_rho

    learned = data['model_learned_params']
    stat_dict['vex'] = initialize_param_from_json(learned['velocity_distribution'])
    stat_dict['area'] = initialize_param_from_json(learned['area_distribution'])

    if data['model_fixed_params']['model_type'] == 'stationary':
        # return info for stationary model
        stat_dict['z'] = np.asarray(z)
        stat_dict['E'] = initialize_param_from_json(learned['entrainment_distribution'])

    elif data['model_fixed_params']['model_type'] == 'crossflow':
        # return info for stationary model
        t = np.asarray(fixed['t'])
        stat_dict['s'] = np.asarray(fixed['s'])
        stat_dict['curfunc'] = CurrMag(**learned['curr_magnitude'])
        stat_dict['headfunc'] = CurrHead(**learned['curr_heading'])
        stat_dict['E'] = (initialize_param_from_json(learned['entrainment_alpha_distribution']),
                          initialize_param_from_json(learned['entrainment_beta_distribution']))

    f.close()

    return stat_dict
