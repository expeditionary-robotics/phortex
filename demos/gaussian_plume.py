'''This script is used to demonstrate the Gaussian Plume model.'''

import matplotlib.pyplot as plt
import numpy as np


def pg_stability_class(wind_speed, day=True, day_type='Strong'):
    '''Method which will return the stability class given the wind-speed, whether it is day or night
    and whether the day is strong/moderate/slight or the night is cloudy/clear
    Input:
    - wind_speed (float) in m/s
    - day (boolean) True is day, False if night
    - day_type (string) if day is True should be one of Strong, moderate or Slight if day is False
    should be one of Cloudy or Clear
    Output:
    - string of stability class'''

    if day is True:
        if day_type not in ['Strong', 'Moderate', 'Slight']:
            print('Only support Strong, Moderate or Slight for daytime characterization')
            return
        else:
            if day_type == 'Strong':
                if wind_speed < 2.:
                    return 'A'
                elif wind_speed >= 2. and wind_speed < 3.:
                    return 'A-B'
                elif wind_speed >= 3. and wind_speed < 5.:
                    return 'B'
                elif wind_speed >= 5.:
                    return 'C'
            elif day_type == 'Moderate':
                if wind_speed < 2.:
                    return 'A-B'
                elif wind_speed >= 2. and wind_speed < 3.:
                    return 'B'
                elif wind_speed >= 3. and wind_speed < 5.:
                    return 'B-C'
                elif wind_speed >= 5. and wind_speed < 6.:
                    return 'C-D'
                elif wind_speed >= 6.:
                    return 'D'
            elif day_type == 'Slight':
                if wind_speed < 2.:
                    return 'B'
                elif wind_speed >= 2. and wind_speed < 5.:
                    return 'C'
                elif wind_speed >= 5.:
                    return 'D'
    if day is False:
        if day_type not in ['Cloudy', 'Clear']:
            print('Only support Clear, or Cloudy for nighttime characterization')
            return
        else:
            if day_type == 'Cloudy':
                if wind_speed < 3.:
                    return 'E'
                elif wind_speed >= 3.:
                    return 'D'
            elif day_type == 'Clear':
                if wind_speed < 3.:
                    return 'F'
                elif wind_speed >= 3. and wind_speed < 5.:
                    return 'E'
                elif wind_speed >= 5.:
                    return 'D'


def briggs_coeffs(stability_class, urban=True):
    '''Method which will return coefficients for calculating the wind vecotr standard deviations
    Input:
    - stability_class (string) one of the classes returned by pg_stability_class
    - urban (boolean) if true it is an urban environment if false it is a rural environment
    output:
    - a, b, c, d, e, f as floats defined by Table 2.2 in Dean thesis'''
    if urban is True:
        if ('A' and 'B' in stability_class) or (stability_class == 'A') or (stability_class == 'B'):
            return 0.32, 0.0004, 0.5, 0.24, 0.0001, -0.5
        elif stability_class == 'C':
            return 0.22, 0.0004, 0.5, 0.2, 0, None
        elif stability_class == 'D':
            return 0.16, 0.0004, 0.5, 0.14, 0.0003, 0.5
        elif ('E' and 'F' in stability_class) or (stability_class == 'E') or (stability_class == 'F'):
            return 0.11, 0.0004, 0.5, 0.08, 0.0015, 0.5
        else:
            print('Only support A-B, C, D, E-F classes')
            return
    elif urban is False:
        if stability_class == 'A':
            return 0.22, 0.0001, 0.5, 0.2, 0, None
        elif stability_class == 'B':
            return 0.16, 0.0001, 0.5, 0.12, 0, None
        elif stability_class == 'C':
            return 0.11, 0.0001, 0.5, 0.08, 0.0002, 0.5
        elif stability_class == 'D':
            return 0.06, 0.0001, 0.5, 0.03, 0.0003, 1
        elif stability_class == 'F':
            return 0.04, 0.0001, 0.5, 0.016, 0.0003, 1
        else:
            print('Only support A, B, C, D, F classes')
            return


def sigma_y(a, b, c, x):
    '''Function for calculating sigma_y, the dispersion parameter
    Input:
    - a, b, c: briggs coefficients based on stability class
    - x: distance downwind
    Output:
    - float or vector of floats that is sigma_y
    '''

    return (a*x) / np.power((1+b*x),c)

def sigma_z(d, e, f, x):
    '''Function for calculating sigma_z, the dispersion parameter
    Input:
    - d, e, f: briggs coefficients based on stability class
    - x: distance downwind
    Output:
    - float or vector of floats that is sigma_z
    '''
    if f is None:
        return d*x
    else:
        return (d*x) / np.power(1 + e*x,f)
