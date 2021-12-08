# coding: utf-8

# author: Brian R. Pauw

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

def extract_feature(h5file, path_to_feature):
    try:
        feature = h5file[path_to_feature][()]
        try:
            units = h5file[path_to_feature].attrs['units']
        except KeyError as err:
            print(err)
            units = ''
    except KeyError as err:
        print(err)
        feature = np.nan
        units = ''
    return feature, units


def test_beam_center(h5file, axis):
    beam_center_axis, units= extract_feature(h5file,'entry1/instrument/detector00/beam_center_'+axis)
    beam_center_axis_std = np.std(beam_center_axis)
    if units == 'm':
        beam_center_axis_std = beam_center_axis_std*10**(6)
    elif units == 'px':
        beam_center_axis_std = beam_center_axis_std*0.0002645833*10**(6)
    else:
        print('the unit is not recognised')
        return False
    if beam_center_axis_std>75:
        return True
    else:
        return False

def test_unstable_instrument(h5file):
    keywords = []
    chamber_pressure,_ =  extract_feature(h5file,'entry1/instrument/chamber_pressure')
    if np.std(chamber_pressure)>0.1:
        keywords.append('ChamberPressureUnstable')

    if test_beam_center(h5file, 'x') or test_beam_center(h5file, 'y'):
        keywords.append('BeamCenterUnstable')
    return keywords
    
def test_source_current(h5file):
    source_current,_ = extract_feature(h5file, 'entry1/instrument/source/current')
    if source_current>25:
        return ['SourceCurrentHigh']
    else:
        return []

def test_source_voltage(h5file):
    source_voltage,_ = extract_feature(h5file, 'entry1/instrument/source/voltage')
    if source_voltage<35:
        return ['SourceVoltgeLow']
    else:
        return []
def test_chamber_pressure(h5file):
    chamber_pressure,_ = extract_feature(h5file, 'entry1/instrument/chamber_pressure')
    if not np.isnan(chamber_pressure).all():
        if (chamber_pressure >1).any():
            return ['ChamberPressureHigh']
        else:
            return []
    else:
        return []

def test_transmission_thickness(h5file):
    tranmission,_ = extract_feature(h5file, 'entry1/sample/transmission')
    if np.mean(tranmission)<0.1:
        return ['TransmissionLow']
    elif np.mean(tranmission) >0.9:
        return ['TransmissionHigh']
    else:
        return []

def test_transmission_stability(h5file, slider = 0.05):
    tranmission,_ = extract_feature(h5file, 'entry1/sample/transmission')
    std = np.std(tranmission)
    if std>(slider*np.mean(tranmission)):
        return ['TransmissionFluctuates']
    else:
        return []
def test_flux_stability(h5file, slider = 0.01):
    flux,_ = extract_feature(h5file, 'entry1/sample/beam/flux')
    std = np.std(flux)
    if std>(slider*np.mean(flux)):
        return ['FluxFluctuates']
    else:
        return []

def add_simple_tags(filename):
    with h5py.File(filename, "r") as h5f:
        return (test_unstable_instrument(h5f)
                +test_source_current(h5f)
                +test_source_voltage(h5f)
                +test_chamber_pressure(h5f)
                +test_transmission_thickness(h5f)
                +test_transmission_stability(h5f)
                +test_flux_stability(h5f))