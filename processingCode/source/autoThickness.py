#!/usr/bin/env python
# coding: UTF-8

"""
Overview
========
This program calculates the thickness from the transmission and X-ray absorption coefficient. 
Uses the "mu" value as stored in the NeXus file, or overridden by command-line option

Required: A python 3.7 installation, numpy, nexusformat, h5py
"""

__author__ = "Brian R. Pauw"
__contact__ = "brian.pauw@bam.de"
__license__ = "GPLv3+"
__date__ = "2020/06/05"
__status__ = "v1"

import numpy as np
from nexusformat import nexus as nx
import argparse  # process input arguments
from pathlib import Path, PureWindowsPath

class autoThickness(object):
    """
    This function calculates the thickness of a sample through the sample transmission. 
    To do this, the total X-ray transmission factor stored in the nexus file is corrected for
    the transmission of the "background" measurement.

    Tested on stacked files first, where the transmission factor is an array. The thickness
    is then calculated as the average finding. 
    """

    filename = None
    backgroundFilename = None
    NeXusXMuPath = None
    NeXusTransmissionPath = None
    NeXusBackgroundPath = None
    NeXusAbsDerivedThicknessPath = None # store the derived thickness in here
    XMu = None # default = -1 as used for structurize, filled in when no absorption coefficient is given in the logbook
    nx = None # nexusformat method instance with filename
    nxBgnd = None # nexusformat method instance with background filename
    test = False # test mode (no write)
    overallSampleTransmission = None
    overallBackgroundTransmisson = None
    thickness = None
    defaultError = 0.01 # if uncertainties cannot be calculated on the transmission, assume this uncertainty
    defaultXMuError = 0.05 # error on the X-ray absorption coefficient

    def __init__(self, **kwargs):
        # process kwargs:
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])
        assert self.filename.exists(), f"Specified filename does not exist: {self.filename}"
        self.nx = nx.nxload(self.filename, 'r+') # read/write, file must exist
        if self.XMu == -1: # not set by command-line option
            self.getNeXusXMu()
        self.getOverallSampleTransmission()
        self.getBackgroundFilename()
        self.getOverallBackgroundTransmission()
        self.calculateThickness()
        if not self.test:
            self.writeThickness()

    def getNeXusXMu(self):
        # obtain X-ray absorption coefficient from stored data:
        if not self.NeXusXMuPath in self.nx:
            print(f'absorption coefficient not stored in sample file: {self.filename}')
            return
        nxXMu = self.nx[self.NeXusXMuPath]
        assert nxXMu.attrs['units'] == '1/m', "X-ray absorption coefficient stored in the NeXus file must be in units of 1/m"
        if nxXMu != -1: # the default when nothing is available:
            self.XMu = float(nxXMu)

    def getOverallTransmission(self, nxObj):
        st = nxObj[self.NeXusTransmissionPath]
        assert st.units=='', "Transmisison units must be in fraction (unitless), not percent or something else"
        st = np.array(st)
        stMean = st.mean()
        if len(st) > 1: stSEM = st.std(ddof = 1) / np.sqrt(len(st))
        else: stSEM = self.defaultError * stMean # cannot estimate, so probably no better than 1%
        return [stMean, stSEM]

    def getOverallSampleTransmission(self):
        self.overallSampleTransmission = self.getOverallTransmission(self.nx)

    def getBackgroundFilename(self):
        bgPath = str(self.nx[self.NeXusBackgroundPath])
        if Path(bgPath).name == str(Path(bgPath)): # not read correctly, probably as windowspath:
            bgPath = Path(PureWindowsPath(bgPath))
        else:
            bgPath = Path(bgPath)
        actualPath = None   
        if bgPath.exists():
            actualPath = bgPath
        elif Path(self.filename.parent, bgPath.name).exists():
            actualPath = Path(self.filename.parent, bgPath.name)
        elif Path(self.filename.parent, bgPath).exists():
            actualPath = Path(self.filename.parent, bgPath)
        else:
            assert False, f"Could not find background file in:\n {bgPath} \n{ Path(self.filename.parent, bgPath.name)} \n  Path(self.filename.parent, bgPath)"
        self.backgroundFilename = actualPath

    def getOverallBackgroundTransmission(self):
        bgNeXus = nx.nxload(self.backgroundFilename, 'r')
        self.overallBackgroundTransmisson = self.getOverallTransmission(bgNeXus)

    def calculateThickness(self):
        assert self.XMu != -1, "X-ray absorption coefficient not set (in 1/m) in sample NeXus file, nor in the command-line arguments. "
        # get sample transmission only:
        print(f'Mean sample transmission: {self.overallSampleTransmission[0]:0.03f} +/- {self.overallSampleTransmission[1]/self.overallSampleTransmission[0]*100:0.02f} % (SEM), Mean background transmission: {self.overallBackgroundTransmisson[0]:0.03f} +/- {self.overallBackgroundTransmisson[1]/self.overallBackgroundTransmisson[0]*100:0.02f} % (SEM), XMu: {self.XMu} 1/m')
        TSample = self.overallSampleTransmission[0]/self.overallBackgroundTransmisson[0]
        TSEMSample = np.sqrt(
            (self.overallSampleTransmission[1]/self.overallSampleTransmission[0])**2 + 
            (self.overallBackgroundTransmisson[1]/self.overallBackgroundTransmisson[0])**2 + 
            self.defaultXMuError**2)
        # calculate using XMu the thickness in m. 
        thickness = -np.log(TSample)/self.XMu
        print(f'Determined sample thickness: {thickness:0.03e} m, uncertainty: {TSEMSample * 100:0.02f} %')
        self.thickness = thickness
        self.thicknessUncertainty = TSEMSample * thickness
    
    def writeThickness(self):
        self.nx[self.NeXusAbsDerivedThicknessPath] = nx.NXfield(
            value = self.thickness, 
            attrs = {
                'units': 'm', 
                'uncertainties': f'{self.NeXusAbsDerivedThicknessPath}_dev',
                'XMu_used': self.XMu
                }
            )
        self.nx[f'{self.NeXusAbsDerivedThicknessPath}_dev'] = nx.NXfield(
            value = self.thicknessUncertainty, 
            attrs = {
                'units': 'm', 
                'uncertainties': f'{self.NeXusAbsDerivedThicknessPath}_dev',
                'XMu_Uncertainty_fraction_used': self.defaultXMuError
                }
            )

def argparser():
    parser = argparse.ArgumentParser(
        description="""
            Calculates the average thickness of a sample for a measurement by the sample transmission and linear X-ray absorption coefficient. \n
            Compensates the overall transmission by the transmission of the background \n
            Example use with command-line-provided X-ray absorption coefficient: \n
            python autoThickness.py testData/20200528_45_expanded_stacked.nxs --XMu 12.3 \n
            Required: A python 3.7 installation
            with: 
                - numpy (for camera reading)
                - pathlib, argparse, nexusformat, h5py
            Programmed by Brian R. Pauw.
            Released under a GPLv3+ license.
            """
    )
    parser.add_argument(
        "filename",
        type=Path,
        default=None,
        help="input filename",
    )
    parser.add_argument(
        "-x", "--XMu", type=float, default=-1, help="X-ray linear absorption coefficient for sample phase in 1/m"
    )
    #NeXusXMuPath
    parser.add_argument(
        "--NeXusXMuPath",
        type=str,
        default='/entry1/sample/mu',
        help="NeXus path to the overall X-ray absorption coefficient of the sample in 1/m",
    )
    parser.add_argument(
        "--NeXusTransmissionPath",
        type=str,
        default='/entry1/sample/transmission',
        help="NeXus path to the measured overall transmission factor",
    )
    parser.add_argument(
        "--NeXusBackgroundPath",
        type=str,
        default='/entry1/backgroundfilename',
        help="NeXus path to the background filename",
    )
    parser.add_argument(
        "--NeXusAbsDerivedThicknessPath",
        type=str,
        default='/entry1/sample/absorptionDerivedThickness',
        help="NeXus path to where the derived, averaged thickness should be stored",
    )
    # image freezes when not downscaled in color!
    parser.add_argument(
        "-t",
        "--test",  
        default = False,
        action="store_true",
        help="test mode, does not write thickness back to NeXus file",
    )
    return parser.parse_args()

if __name__ == "__main__":
    # process input arguments
    adict = argparser()
    # I want a kwargs object, so convert args:
    adict = vars(adict)
    autoThickness(**adict)  # and expand to kwargs
