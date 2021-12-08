# coding: utf-8

# author: Brian R. Pauw, Glen J. Smales
# date: 2019.08.26

# Structurizes the files we get from the MAUS into a proper NeXus/NXsas file.
# based on the 3.8 version of the structurize notebook.
# V13 also stores a list of structurized files that can be uploaded to the database
# v3p16 works for newer files only, after a change in measurement strategy
# changes in structure: every measurement now has its own beam profile, and a second beam profile with the sample in place. 
# these could be used to calculate a second measure for the transmission factor and flux as well in due time. 
# for now we just store them alongside the data. 
# ==

# we need some libraries to run these things.
# %load_ext blackcellmagic
import hdf5plugin
import pandas, scipy
import numpy as np
import h5py
import sys, os
import glob
from io import StringIO
from pathlib import Path
from SAXSClasses import readLog  # reads our logbooks
from SAXSClasses import NXUpdate  # can update single entries in NeXus files
from SAXSClasses import H5Base  # base class for writing HDF5 files
import argparse
import multiprocessing

# for the beam finding:
from skimage import filters
from skimage.measure import regionprops

NeXpandVersion = "3.16"

# set up the argument parser method:
def argparser():
    parser = argparse.ArgumentParser(
        description="""
        A script for Structurizes the files we get from the MAUS into a proper NeXus/NXsas file.
        based on the 3.8 version of the structurize notebook.
        
        example usage:
        python Structurize_v3p9.py -i logbooks\Logbook_MAUS.xls -b 20190806
        """
    )
    parser.add_argument(
        "-i",
        "--inputFile",
        type=str,
        required=True,
        help="Input excel measurement logbook",
    )
    parser.add_argument(
        "-b",
        "--baseDate",
        type=str,
        required=True,
        help="(Optionally partial) YYYYMMDD date containing the files to structurize ",
    )
    parser.add_argument(
        "-f",
        "--filenum",
        type=int,
        required=False,
        default = -1,
        help="(Optional) file number of the file to process",
    )
    return parser.parse_args()

def analyzeBeam(directfile, imageType = 'direct', samplefile = None):
    assert isinstance(directfile, Path)
    assert directfile.is_file()
    assert imageType in ['direct', 'sample', 'both']
    if (imageType == 'sample') or (imageType == 'both'):
        assert isinstance(samplefile, Path), 'for sample-transmitted beam analysis, a direct-beam file and a sample-transmitted beam file must be provided!'

    with h5py.File(directfile, "r") as h5r:
        imageData = h5r['/entry1/instrument/detector00/data'][()]
        recordingTime = h5r['/entry1/instrument/detector00/count_time'][()]

    # label the main beam feature for calculation, this could be powerful and cool. 
    twoDImage = imageData[0, :, :]

    # get rid of masked or pegged pixels
    labeled_foreground = (np.logical_and(twoDImage >= 0, twoDImage <= 1e6)).astype(int)
    maskedTwoDImage = twoDImage * labeled_foreground # apply mask
    threshold_value = np.maximum(1, 0.0001 * maskedTwoDImage.max()) # filters.threshold_otsu(maskedTwoDImage) # ignore zero pixels
    labeled_peak = (maskedTwoDImage > threshold_value).astype(int) # label peak
    properties = regionprops(labeled_peak, twoDImage) # calculate region properties
    center_of_mass = properties[0].centroid # center of mass (unweighted by intensity)
    weighted_center_of_mass = properties[0].weighted_centroid # center of mass (weighted)

    ITotal_region = np.sum(maskedTwoDImage[
            np.maximum(int(weighted_center_of_mass[0] - 25), 0): np.minimum(int(weighted_center_of_mass[0] + 25), maskedTwoDImage.shape[0]), 
            np.maximum(int(weighted_center_of_mass[1] - 25), 0): np.minimum(int(weighted_center_of_mass[1] + 25), maskedTwoDImage.shape[1])
        ])

    print(center_of_mass)
    print(ITotal_region)

    # write back:..
    if (imageType == 'direct') or (imageType == 'both'):
        NXUpdate(directfile, NeXusPath = '/entry1/sample/beam/flux', updateValue=ITotal_region/recordingTime)
        # Coordinate mismatch by half a pixel between dawn and Python. Python pixels have pixel coordinates in the center, dawn has them at the edges. 
        # the first python pixel extends from -0.5 to 0.5, in DAWN it goes from 0 to 1. 
        NXUpdate(directfile, NeXusPath = '/entry1/instrument/detector00/beam_center_x', updateValue=weighted_center_of_mass[1] + 0.5) 
        NXUpdate(directfile, NeXusPath = '/entry1/instrument/detector00/beam_center_y', updateValue=weighted_center_of_mass[0] + 0.5) # ibid
        for destloc in ['/entry1/instrument/detector00/beam_center_x', '/entry1/instrument/detector00/beam_center_y']:
            with h5py.File(directfile, 'a') as h5f:
                h5f[destloc].attrs['units'] = 'px'

    if (imageType == 'sample') or (imageType == 'both'):
        # integrate intensity using the same coordinates as the direct beam:
        with h5py.File(samplefile, "r") as h5r:
            imageData = h5r['/entry1/instrument/detector00/data'][()]
            recordingTimeSample = h5r['/entry1/instrument/detector00/count_time'][()]
        twoDImage = imageData[0, :, :]
        # get rid of masked or pegged pixels
        labeled_foreground = (np.logical_and(twoDImage >= 0, twoDImage <= 1e6)).astype(int)
        maskedTwoDImage = twoDImage * labeled_foreground # apply mask
        ITotal_region_sample = np.sum(maskedTwoDImage[
            np.maximum(int(weighted_center_of_mass[0] - 25), 0): np.minimum(int(weighted_center_of_mass[0] + 25), maskedTwoDImage.shape[0]), 
            np.maximum(int(weighted_center_of_mass[1] - 25), 0): np.minimum(int(weighted_center_of_mass[1] + 25), maskedTwoDImage.shape[1])
        ])

        # read information from direct beam file:
        with h5py.File(directfile, "r") as h5r, h5py.File(samplefile, 'a') as h5f:
            directFlux = h5r['/entry1/sample/beam/flux'][()]
            for destloc in [
                '/entry1/instrument/detector00/beam_center_x', 
                '/entry1/instrument/detector00/beam_center_y', 
                '/entry1/sample/beam/flux']:
                h5f[destloc][...] = h5r[destloc][()]
                h5f[destloc].attrs.update(h5r[destloc].attrs)

        # NXUpdate(filename, NeXusPath = '/entry1/sample/beam/flux', updateValue=[directFlux])
        NXUpdate(samplefile, NeXusPath = '/entry1/sample/transmission', updateValue=(ITotal_region_sample/recordingTimeSample) / directFlux)

# Now add this information back into the measurement file
def beamInfoToSampleMeas(samplefile, directbeamfile, samplebeamfile):
    """ moves the beam flux, beam center, sample transmission information from their respective beam measurement files to the sample measurement file"""
    assert isinstance(samplefile, Path)
    assert isinstance(directbeamfile, Path)
    assert isinstance(samplebeamfile, Path)
    # read information from direct beam file:
    with h5py.File(directbeamfile, "r") as h5r, h5py.File(samplefile, 'a') as h5f:
        for destloc in [
            '/entry1/instrument/detector00/beam_center_x', 
            '/entry1/instrument/detector00/beam_center_y', 
            '/entry1/sample/beam/flux']:
            h5f[destloc][...] = h5r[destloc][()]
            h5f[destloc].attrs.update(h5r[destloc].attrs)
        # add direct beam image
        h5f["/entry1/instrument/detector00/direct_beam_image"][
            ...
        ] = h5r["/entry1/instrument/detector00/data"][()]

    with h5py.File(samplebeamfile, "r") as h5r, h5py.File(samplefile, 'a') as h5f:
            h5f['/entry1/sample/transmission'][...] = h5r['/entry1/sample/transmission'][()]
            h5f['/entry1/sample/transmission'].attrs.update(h5r['/entry1/sample/transmission'].attrs)
            # add sample_transmitted beam image:
            h5f["/entry1/instrument/detector00/sample_beam_image"][
                ...
            ] = h5r["/entry1/instrument/detector00/data"][()]

# set up script to add structure to the existing MOUSE files.

class addSaxslab(H5Base):
    """
    The basic NeXus files that come out of the MAUS have some information in them, 
    but are not structured in a way DAWN can fully exploit the metadata. 
    This script takes the input file, and writes an output file with an expanded structure
    """

    pixelsize = 0.075e-3  # m
    dataDict = {}
    # copying whole groups:
    groupDict = {"/entry1": "/"}#, "/saxs/Saxslab": "/"}
    addAttributeDict = {}
    pruneList = [  # these items will be deleted from the output file
        "/entry1/sample/psi-rotation",
        "/entry1/sample/x-translation",
        "/entry1/sample/z-translation",
        "/entry1/instrument/detector00/distance",
    ]

    def initDataDict(self):
        """ 
        Here, we define what should be in the output file,
        and what values in the input file they should be based on.
        syntax: 
         "srcloc" : source data (hdf5) location 
         "destloc" : destination dataset (hdf5) location , 
         "attributes" : additional attributes as you like in key-value pairs.
         "datatype" : will attempt to cast into datatype
         "default": must be None if no default value specified
         "ndmin": set this to minimum number of dimensions for output array if required, or leave None
         "lambfunc": a lambda function to apply to the retrieved value before storing it in the NeXus file
        """
        self.dataDict = pandas.DataFrame(
            columns=["srcloc", "destloc", "attributes", "datatype", "default", "lambfunc"]
        )

        # now we start appending information:
        appendList = [
            {
                # "srcloc": "/saxs/Saxslab/det_exposure_time",
                "srcloc": "/entry1/instrument/detector00/count_time",
                "destloc": "/entry1/instrument/detector00/count_time",
                "attributes": {"units": "s"},
                "datatype": np.float64,
                "default": 1,
                "ndmin": 1,
            },
            {
                "srcloc": "/entry1/instrument/detector00/beam_center_x",
                "destloc": "/entry1/instrument/detector00/beam_center_y",
                "attributes": {"units": "m"},
                "datatype": np.float64,
                "default": None,
                "ndmin": 1,
                "lambfunc": lambda x: x * self.pixelsize,
            },
            {
                "srcloc": "/entry1/instrument/detector00/beam_center_y",
                "destloc": "/entry1/instrument/detector00/beam_center_x",
                "attributes": {"units": "m"},
                "datatype": np.float64,
                "default": None,
                "ndmin": 1,
                "lambfunc": lambda x: x * self.pixelsize,
            },
            {
                "srcloc": "/entry1/instrument/detector00/data",
                "destloc": "/entry1/instrument/detector00/data",
                "attributes": {"units": "counts"},
                "datatype": np.float64,
                "default": None,
                "ndmin": 3,
                "lambfunc": lambda x: np.squeeze(x, axis=0),
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/detector00/transformations/euler_c",
                "attributes": {
                    "depends_on": "./euler_b",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "mm",
                    "transformation_type": "rotation",
                    "units": "deg",
                    "vector": [0.0, 0.0, 1.0],
                },
                "datatype": np.float64,
                "default": -180.0,
                "ndmin": 1,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/detector00/transformations/euler_b",
                "attributes": {
                    "depends_on": "./euler_a",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "mm",
                    "transformation_type": "rotation",
                    "units": "deg",
                    "vector": [0.0, 1.0, 0.0],
                },
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/detector00/transformations/euler_a",
                "attributes": {
                    "depends_on": "./det_y",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "mm",
                    "transformation_type": "rotation",
                    "units": "deg",
                    "vector": [0.0, 0.0, 1.0],
                },
                "datatype": np.float64,
                "default": 180.0,
                "ndmin": 1,
            },
            {
                "srcloc": "/entry1/instrument/detector00/beam_center_y",
                "destloc": "/entry1/instrument/detector00/transformations/det_y",
                "attributes": {
                    "depends_on": "./det_z",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "mm",
                    "transformation_type": "translation",
                    "units": "m",
                    "vector": [1, 0.0, 0.0],
                },
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
                "lambfunc": lambda x: x * self.pixelsize,
            },
            {
                "srcloc": "/entry1/instrument/detector00/beam_center_x",
                "destloc": "/entry1/instrument/detector00/transformations/det_z",
                "attributes": {
                    "depends_on": "./det_x",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "mm",
                    "transformation_type": "translation",
                    "units": "m",
                    "vector": [0.0, 1, 0.0],
                },
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
                "lambfunc": lambda x: x * self.pixelsize,
            },
            # future: det_x transformation replaced by the encoder read-out rather than the set distance. 
            {
                "srcloc": "/entry1/instrument/detector00/distance",
                "destloc": "/entry1/instrument/detector00/transformations/det_x",
                "attributes": {
                    "depends_on": ".",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "mm",
                    "transformation_type": "translation",
                    "units": "mm",
                    "vector": [0.0, 0.0, 1.0],
                },
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
            },
            {
                "srcloc": "/entry1/instrument/detector00/x_pixel_size",
                "destloc": "/entry1/instrument/detector00/detector_module/fast_pixel_direction",
                "attributes": {
                    "depends_on": "./module_offset",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "m",
                    "transformation_type": "translation",
                    "units": "m",
                    "vector": [-1.0, 0.0, 0.0],
                },
                "datatype": np.float64,
                "default": self.pixelsize,
                "ndmin": 1,
            },
            {
                "srcloc": "/entry1/instrument/detector00/y_pixel_size",
                "destloc": "/entry1/instrument/detector00/detector_module/slow_pixel_direction",
                "attributes": {
                    "depends_on": "./module_offset",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "m",
                    "transformation_type": "translation",
                    "units": "m",
                    "vector": [0.0, -1.0, 0.0],
                },
                "datatype": np.float64,
                "default": self.pixelsize,
                "ndmin": 1,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/detector00/detector_module/module_offset",
                "attributes": {
                    "depends_on": "../transformations/euler_c",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "m",
                    "transformation_type": "translation",
                    "units": "m",
                    "vector": [0.0, 0.0, 0.0],
                },
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/detector00/detector_module/data_origin",
                "attributes": {},
                "datatype": np.int32,
                "default": [0, 0],
                "ndmin": 0,
            },  # these should not be stacked!
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/detector00/detector_module/data_size",
                "attributes": {},
                "datatype": np.int32,
                "default": [1030, 1065],
                "ndmin": 0,
            },  # these should not be stacked!
            # beam information (at sample)
            {
                "srcloc": "/saxs/Saxslab/wavelength",
                "destloc": "/entry1/sample/beam/incident_wavelength",
                "attributes": {"units": "angstrom"},
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
            },
            # Chamber pressure
            {
                "srcloc": "/saxs/Saxslab/chamber_pressure",
                "destloc": "/entry1/instrument/chamber_pressure",
                "attributes": {"units": "mbar"},
                "datatype": np.float64,
                "default": 1.0,
                "ndmin": 1,
            },
            # interferometer strip location
            {
                "srcloc": "/saxs/Saxslab/detx_enc",
                "destloc": "/entry1/instrument/detector00/det_x_encoder",
                "attributes": {
                    "units": "mm", 
                    "transformation_type": "translation", 
                    "offset": 0, 
                    "offset_units": "mm", 
                    "vector":[0, 0, 1], 
                    "depends_on": ".",
                    "description": "Independent Renishaw encoder strip next to detector carriage",
                    },
                "datatype": np.float64,
                "default": -1.0,
                "ndmin": 1,
            },
            # Future code: set interferometer as the detector transformation x with offset of:
            # {
            #     "srcloc": "/entry1/instrument/detector00/det_x_encoder",
            #     "destloc": "/entry1/instrument/detector00/transformations/det_x",
            #     "attributes": {
            #         "depends_on": ".",
            #         "offset": [0.0, 0.0, 0.0],
            #         "offset_units": "mm",
            #         "transformation_type": "translation",
            #         "units": "mm",
            #         "vector": [0.0, 0.0, 1.0],
            #     },
            #     "datatype": np.float64,
            #     "default": 0.0,
            #     "ndmin": 1,
            #     "lambfunc": lambda x: x -28.9868,
            # },
            # sample information
            {
                "srcloc": "/saxs/Saxslab/saxsconf_Izero",
                "destloc": "/entry1/sample/beam/flux",
                "attributes": {"units": "s-1"},
                "datatype": np.float64,
                "default": 1.0,
                "ndmin": 1,
            },
            {
                "srcloc": "/saxs/Saxslab/sample_transfact",
                "destloc": "/entry1/sample/transmission",
                "attributes": {"units": ""},
                "datatype": np.float64,
                "default": 1.0,
                "ndmin": 1,
            },
            {
                "srcloc": "/saxs/Saxslab/sample_zpos",
                "destloc": "/entry1/sample/transformations/sample_z",
                "attributes": {
                    "depends_on": "./sample_x",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "mm",
                    "transformation_type": "translation",
                    "units": "mm",
                    "vector": [0.0, 1.0, 0.0],
                },
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
            },
            {
                "srcloc": "/saxs/Saxslab/sample_ypos",
                "destloc": "/entry1/sample/transformations/sample_y",
                "attributes": {
                    "depends_on": "./sample_z",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "mm",
                    "transformation_type": "translation",
                    "units": "mm",
                    "vector": [1.0, 0.0, 0.0],
                },
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/sample/transformations/sample_x",
                "attributes": {
                    "depends_on": ".",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "mm",
                    "transformation_type": "translation",
                    "units": "mm",
                    "vector": [0.0, 0.0, 1.0],
                },
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
            },
            {
                "srcloc": "/saxs/Saxslab/sample_thickness",
                "destloc": "/entry1/sample/thickness",
                "attributes": {"units": "m"},
                "datatype": np.float64,
                "default": 1.0,
                "ndmin": 1,
                "lambfunc": lambda x: x / 100,
            },  # stored in cm in the SAXSLab file.
            # beamstop information
            {
                "srcloc": "/saxs/Saxslab/bsh",
                "destloc": "/entry1/instrument/beamstop/transformations/bs_y",
                "attributes": {
                    "depends_on": "./bs_z",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "mm",
                    "transformation_type": "translation",
                    "units": "mm",
                    "vector": [1.0, 0.0, 0.0],
                },
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
            },
            {
                "srcloc": "/saxs/Saxslab/bsv",
                "destloc": "/entry1/instrument/beamstop/transformations/bs_z",
                "attributes": {
                    "depends_on": "../../detector/transformations/det_x",
                    "offset": [0.0, 0.0, 0.0],
                    "offset_units": "mm",
                    "transformation_type": "translation",
                    "units": "mm",
                    "vector": [0.0, 1.0, 0.0],
                },
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/beamstop/description",
                "attributes": {},
                "datatype": str,
                "default": "circular",
                "ndmin": None,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/beamstop/size",
                "attributes": {"units": "mm"},
                "datatype": np.float64,
                "default": 2.0,
                "ndmin": None,
            },
            # source information
            {
                "srcloc": "/saxs/Saxslab/wavelength",
                "destloc": "/entry1/instrument/monochromator/wavelength",
                "attributes": {"units": "angstrom"},
                "datatype": np.float64,
                "default": 0.0,
                "ndmin": 1,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/monochromator/wavelength_error",
                "attributes": {"units": "angstrom"},
                "datatype": np.float64,
                "default": 0.005,  # estimated
                "ndmin": 1,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/monochromator/crystal/usage",
                "attributes": {},
                "datatype": str,
                "default": "Bragg",  # estimated
                "ndmin": None,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/monochromator/crystal/type",
                "attributes": {},
                "datatype": str,
                "default": "Multilayer",  # estimated
                "ndmin": None,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/monochromator/crystal/order_no",
                "attributes": {},
                "datatype": np.int32,
                "default": 1,  # estimated
                "ndmin": None,
            },
            {
                "srcloc": "/saxs/Saxslab/wavelength",
                "destloc": "/entry1/instrument/monochromator/crystal/wavelength",
                "attributes": {"units": "angstrom"},
                "datatype": np.float32,
                "default": 1,  # estimated
                "ndmin": 1,
            },
            # source:
            {
                "srcloc": "/saxs/Saxslab/source_type",
                "destloc": "/entry1/instrument/source/name",
                "attributes": {},
                "datatype": str,
                "default": "Xenocs microfocus source",  # estimated
                "ndmin": None,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/source/type",
                "attributes": {},
                "datatype": str,
                "default": "Fixed Tube X-ray",  # estimated
                "ndmin": None,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/source/probe",
                "attributes": {},
                "datatype": str,
                "default": "X-ray",  # estimated
                "ndmin": None,
            },
            {
                "srcloc": "/saxs/Saxslab/source_ma",
                "destloc": "/entry1/instrument/source/current",
                "attributes": {"units": "mA"},
                "datatype": np.float64,
                "default": 0.0,  # estimated
                "ndmin": 1,
            },
            {
                "srcloc": "/saxs/Saxslab/source_kV",
                "destloc": "/entry1/instrument/source/voltage",
                "attributes": {"units": "kV"},
                "datatype": np.float64,
                "default": 0.0,  # estimated
                "ndmin": 1,
            },
            # 3.4 additions:
            {
                "srcloc": None,
                "destloc": "/entry1/sample/sampleholder",
                "attributes": {},
                "datatype": str,
                "default": "",  # estimated
                "ndmin": None,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/instrument/configuration",
                "attributes": {},
                "datatype": np.int32,
                "default": 1,  # estimated
                "ndmin": 1,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/sample/sampleowner",
                "attributes": {},
                "datatype": str,
                "default": "",  # estimated
                "ndmin": None,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/user",
                "attributes": {},
                "datatype": str,
                "default": "",  # estimated
                "ndmin": None,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/proposal/number",
                "attributes": {},
                "datatype": np.int32,
                "default": 0,  # estimated
                "ndmin": None,
            },
            # 3.6 additions
            {
                "srcloc": None,
                "destloc": "/entry1/backgroundfilename",
                "attributes": {},
                "datatype": str,
                "default": "",  # estimated
                "ndmin": None,
            },
            {
                "srcloc": "/entry1/instrument/detector00/data",
                "destloc": "/entry1/instrument/detector00/direct_beam_image",
                "attributes": {"units": "counts"},
                "datatype": np.float64,
                "default": None,
                "ndmin": 3,
                "lambfunc": lambda x: np.squeeze(x * 0, axis=0),
            },  # zero out
            {
                "srcloc": "/entry1/instrument/detector00/data",
                "destloc": "/entry1/instrument/detector00/sample_beam_image",
                "attributes": {"units": "counts"},
                "datatype": np.float64,
                "default": None,
                "ndmin": 3,
                "lambfunc": lambda x: np.squeeze(x * 0, axis=0),
            },  # zero out
            # 3.7 addition
            {
                "srcloc": None,
                "destloc": "/entry1/maskfilename",
                "attributes": {},
                "datatype": str,
                "default": "",  # estimated
                "ndmin": None,
            },
            # 3.14 addition
            {
                "srcloc": None,
                "destloc": "/entry1/dispersantbackgroundfilename",
                "attributes": {},
                "datatype": str,
                "default": "",  # estimated
                "ndmin": None,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/sample/mu",
                "attributes": {"units": "1/m"},
                "datatype": str,
                "default": "",  # estimated
                "ndmin": None,
            },
            {
                "srcloc": None,
                "destloc": "/entry1/processingpipeline",
                "attributes": {},
                "datatype": str,
                "default": "",  
                "ndmin": None,
            },
        ]
        
        for item in appendList:
            self.dataDict = self.dataDict.append(item, ignore_index=True)

        self.addAttributeDict = {
            # /
            "/@default": "entry1",
            "/@NeXpand_version": self.NeXpandScriptVersion,
            # /entry1
            "/entry1@NX_class": "NXentry",
            "/entry1@default": "instrument",
            "/entry1/instrument@default": "detector00",
            # /entry1/instrument/detector00
            "/entry1/instrument/detector00@NX_class": "NXdetector",
            "/entry1/instrument/detector00@default": "data",
            "/entry1/instrument/detector00@depends_on": "./transformations/euler_c",
            "/entry1/instrument/detector00/detector_module@NX_class": "NXdetector_module",
            "/entry1/instrument/detector00/transformations@NX_class": "NXtransformations",
            "/entry1/instrument/beamstop@NX_class": "NXbeam_stop",
            "/entry1/instrument/beamstop@depends_on": "./transformations_bs_y",
            "/entry1/instrument/beamstop/transformations@NX_class": "NXtransformations",
            # /entry1/sample
            "/entry1/sample@NX_class": "NXsample",
            "/entry1/sample@depends_on": "./transformations/sample_y",
            "/entry1/sample/beam@NX_class": "NXbeam",
            "/entry1/sample/transformations@NX_class": "NXtransformations",
            # /entry1/instrument
            "/entry1/instrument@NX_class": "NXinstrument",
            "/entry1/instrument/monochromator@NX_class": "NXmonochromator",
            "/entry1/instrument/monochromator/crystal@NX_class": "NXcrystal",
            "/entry1/instrument/source@NX_class": "NXsource",
        }


# set up script to add structure to the existing MAUS files.
class addDectris(H5Base):
    """
    Adds the extra metadata from the Dectris HDF5 files to a previously generated measurement file
    """

    dataDict = {}
    # copying whole groups:
    groupDict = {
        "/entry/data": "/entry1/frames/",
        "/entry/instrument/detector/detectorSpecific": "/entry1/instrument/detector00/",
    }
    addAttributeDict = {}
    pruneList = []  # these items will be deleted from the output file

    def initDataDict(self):
        """ 
        Here, we define what should be in the output file,
        and what values in the input file they should be based on.
        syntax: 
         "srcloc" : source data (hdf5) location 
         "destloc" : destination dataset (hdf5) location , 
         "attributes" : additional attributes as you like in key-value pairs.
         "datatype" : will attempt to cast into datatype
         "default": must be None if no default value specified
         "ndmin": set this to minimum number of dimensions for output array if required, or leave None
         "lambfunc": a lambda function to apply to the retrieved value before storing it in the NeXus file
        """
        self.dataDict = pandas.DataFrame(
            columns=["srcloc", "destloc", "attributes", "datatype", "default", "lambfunc"]
        )

        # now we start appending information:
        appendList = [
            {
                "srcloc": "/entry/instrument/detector/bit_depth_image",
                "destloc": "/entry1/instrument/detector00/bit_depth_image",
                "attributes": {},
                "datatype": np.int32,
                "default": 0,  # estimated
                "ndmin": None,
            },
            {
                "srcloc": "/entry/instrument/detector/bit_depth_readout",
                "destloc": "/entry1/instrument/detector00/bit_depth_readout",
                "attributes": {},
                "datatype": np.int32,
                "default": 0,  # estimated
                "ndmin": None,
            },
            {
                "srcloc": "/entry/instrument/detector/count_time",
                "destloc": "/entry1/frames/count_time",
                "attributes": {},  # existing attributes such as units are automatically copied.
                "datatype": np.float32,
                "default": 0,
                "ndmin": 1,
            },
            {
                "srcloc": "/entry/instrument/detector/countrate_correction_applied",
                "destloc": "/entry1/instrument/detector00/countrate_correction_applied",
                "attributes": {},
                "datatype": np.int32,
                "default": 0,
                "ndmin": None,
            },
            {
                "srcloc": "/entry/instrument/detector/efficiency_correction_applied",
                "destloc": "/entry1/instrument/detector00/efficiency_correction_applied",
                "attributes": {},
                "datatype": np.int32,
                "default": 0,
                "ndmin": None,
            },
            {
                "srcloc": "/entry/instrument/detector/flatfield_correction_applied",
                "destloc": "/entry1/instrument/detector00/flatfield_correction_applied",
                "attributes": {},
                "datatype": np.int32,
                "default": 0,
                "ndmin": None,
            },
            {
                "srcloc": "/entry/instrument/detector/pixel_mask_applied",
                "destloc": "/entry1/instrument/detector00/pixel_mask_applied",
                "attributes": {},
                "datatype": np.int32,
                "default": 0,
                "ndmin": None,
            },
            {
                "srcloc": "/entry/instrument/detector/virtual_pixel_correction_applied",
                "destloc": "/entry1/instrument/detector00/virtual_pixel_correction_applied",
                "attributes": {},
                "datatype": np.int32,
                "default": 0,
                "ndmin": None,
            },
            {
                "srcloc": "/entry/instrument/detector/description",
                "destloc": "/entry1/instrument/detector00/description",
                "attributes": {},
                "datatype": str,
                "default": "",
                "ndmin": None,
            },
            {
                "srcloc": "/entry/instrument/detector/detector_number",
                "destloc": "/entry1/instrument/detector00/detector_number",
                "attributes": {},
                "datatype": str,
                "default": "",
                "ndmin": None,
            },
            {
                "srcloc": "/entry/instrument/detector/detector_readout_time",
                "destloc": "/entry1/instrument/detector00/detector_readout_time",
                "attributes": {},  # existing attributes such as units are automatically copied.
                "datatype": np.float32,
                "default": "",
                "ndmin": 1,
            },
            {
                "srcloc": "/entry/instrument/detector/sensor_material",
                "destloc": "/entry1/instrument/detector00/sensor_material",
                "attributes": {},
                "datatype": str,
                "default": "Si",
                "ndmin": None,
            },
            {
                "srcloc": "/entry/instrument/detector/sensor_thickness",
                "destloc": "/entry1/instrument/detector00/sensor_thickness",
                "attributes": {},
                "datatype": np.float32,
                "default": "",
                "ndmin": 1,
            },
            {
                "srcloc": "/entry/instrument/detector/threshold_energy",
                "destloc": "/entry1/instrument/detector00/threshold_energy",
                "attributes": {},
                "datatype": np.float32,
                "default": "",
                "ndmin": 1,
            },
            {
                "srcloc": "/entry/instrument/detector/x_pixel_size",
                "destloc": "/entry1/instrument/detector00/x_pixel_size",
                "attributes": {},
                "datatype": np.float32,
                "default": "",
                "ndmin": 1,
            },
            {
                "srcloc": "/entry/instrument/detector/y_pixel_size",
                "destloc": "/entry1/instrument/detector00/y_pixel_size",
                "attributes": {},
                "datatype": np.float32,
                "default": "",
                "ndmin": 1,
            },
            # try one more:
            # {
            #     "srcloc": '/entry1/instrument/detector00/detectorSpecific/data_collection_date',
            #     "destloc": "/entry1/instrument/startTimeStamp",
            #     "attributes": {},
            #     "datatype": str,
            #     "default": "",
            #     "ndmin": None,
            #     "lambfunc": lambda x: bytes(x, encoding='utf-8').decode('utf-8'),
            # },
        ]

        for item in appendList:
            self.dataDict = self.dataDict.append(item, ignore_index=True)

        self.addAttributeDict = {}

# this is where the SAXSlab nexus files are converted in DAWN-compatible NeXus
def NeXify(ifname, ofname, NeXpandVersion=None):
    dectrisFilename = None
    # try to get the dectris data file, these sometimes don't get transferred along
    try:
        dectrisFilename = sorted(ifname.parent.glob("*master*"))[0]
    except IndexError:
        print("Warning: Dectris Master file not available!")
    except:
        raise

    print("# # # # # Working on file: {}".format(ifname))
    # expand the single frames into more detailed frames
    n5o = addSaxslab(
        ifname, ofname, NeXpandVersion=NeXpandVersion, ifExist="check"
    )  # replace is also an option. Forces replacement
    if dectrisFilename is not None:
        n5o = addDectris(
            dectrisFilename,
            ofname,
            NeXpandVersion=NeXpandVersion,
            ifExist="concatenate",
        )
        # add hasDectris field
        n5o.addAttribute("/entry1/instrument/detector00@hasDectrisDataFile", True)
    else:
        n5o.addAttribute("/entry1/instrument/detector00@hasDectrisDataFile", False)
    # mixin direct beam if available
    # no: cannot add direct beam here since it's ill-defined where it belongs to. Must be done in stacker.

def sortkey(item):
    return int(str(item.parent).split("_")[-1])

class NXConcat(H5Base):
    """
    Concatenates multiple NeXus files *with identical structure* to frames of a single file. 
    Useful for combining multiple exposures, structurized using NeXpand, into a series of frames.
    
    All data in a single array in the input nexus files are concatenated along axis NeXusAxis. 
    Non-array values are read from the first file and stored in the new file. 
    """

    inflist = None
    ofname = None
    concatAxis = 1
    remove = True
    bDict = {}
    allItems = []  # is filled in by self.listStructure
    forceDType = {"entry1/frames/data/data_000001": np.float64}
    stackItems = [
        # 'entry1/frames/data/data_000001',
        "entry1/instrument/detector00/data",
        "entry1/instrument/detector00/direct_beam_image",
        "entry1/instrument/detector00/sample_beam_image",
        "entry1/instrument/detector00/count_time",
        "entry1/instrument/detector00/beam_center_y",
        "entry1/instrument/detector00/beam_center_x",
        "entry1/instrument/detector00/x_pixel_size",
        "entry1/instrument/detector00/y_pixel_size",
        "entry1/instrument/detector00/threshold_energy",
        "entry1/instrument/detector00/sensor_thickness",
        "entry1/instrument/detector00/transformations/euler_c",
        "entry1/instrument/detector00/transformations/euler_b",
        "entry1/instrument/detector00/transformations/euler_a",
        "entry1/instrument/detector00/transformations/det_y",
        "entry1/instrument/detector00/transformations/det_z",
        "entry1/instrument/detector00/transformations/det_x",
        "entry1/instrument/detector00/detector_module/fast_pixel_direction",
        "entry1/instrument/detector00/detector_module/slow_pixel_direction",
        "entry1/instrument/detector00/detector_module/module_offset",
        # 'entry1/instrument/detector00/detector_module/data_origin', # these should not be stacked!
        # 'entry1/instrument/detector00/detector_module/data_size', # these should not be stacked!
        "entry1/sample/transformations/sample_z",
        "entry1/sample/transformations/sample_y",
        "entry1/sample/transformations/sample_x",
        "entry1/sample/transmission",
        "entry1/sample/thickness",
        "entry1/sample/beam/flux",
        "entry1/sample/beam/incident_wavelength",
        "entry1/instrument/monochromator/crystal/wavelength",
        "entry1/instrument/monochromator/wavelength_error",
        "entry1/instrument/monochromator/wavelength",
        "entry1/instrument/beamstop/transformations/bs_y",
        "entry1/instrument/beamstop/transformations/bs_z",
        "entry1/instrument/chamber_pressure",
        "entry1/instrument/detector00/det_x_encoder",
        # '/entry1/instrument/detector00/detectorSpecific/data_collection_date',
    ]

    def clear(self):
        self.inflist = []
        self.ofname = None
        self.concatAxis = 1
        self.remove = True
        self.bDict = {}
        self.allItems = []

    def __init__(
        self,
        inflist=[],
        ofname=None,
        concatAxis=1,
        remove=False,
        NeXpandScriptVersion=None,
    ):
        self.clear()
        self.inflist = inflist
        self.ofname = ofname
        self.concatAxis = concatAxis
        self.remove = remove
        self.NeXpandScriptVersion = NeXpandScriptVersion

        # delete output file if exists, and requested by input flag
        if os.path.exists(self.ofname) and self.remove:
            print("file {} exists, removing...".format(self.ofname))
            os.remove(self.ofname)

        for infile in self.inflist:
            self.expandBDict(infile)

        self.listStructure(self.inflist[0])
        # remove stackItems from that list
        for x in self.stackItems:
            # print("removing item {} from self.allitems: {} \n \n".format(x, self.allItems))
            try:
                self.allItems.remove(x)
            except ValueError:
                print("Item {} not found in allItems, skipping...".format(x))

        self.hdfDeepCopy(self.inflist[0], self.ofname)
        with h5py.File(self.ofname, "a") as h5f:
            # del h5f["Saxslab"] not the issue here.
            h5f["/"].attrs["default"] = "entry1"

    def _stackIt(self, name, obj):
        # print(name)
        if name in self.stackItems:
            if name in self.bDict:  # already exists:
                try:
                    # self.bDict[name] = np.concatenate((self.bDict[name], obj[()]), axis = self.concatAxis)
                    self.bDict[name].append(
                        obj[()]
                    )  # np.stack((self.bDict[name], obj[()]))

                except ValueError:
                    print(
                        "\n\n file: {} \n NeXus path: {} \n value: {}".format(
                            self.ofname, name, obj[()]
                        )
                    )
                    print(
                        "\n\n bDict[name]: {} \n bDict[name] shape: {} \n value shape: {}".format(
                            self.bDict[name], self.bDict[name].shape, obj[()].shape
                        )
                    )
                    raise
                except:
                    raise
            else:  # create entry
                self.bDict[name] = [obj[()]]  # list of entries

    def expandBDict(self, filename):

        with h5py.File(filename, "r") as h5f:

            h5f.visititems(self._stackIt)

    def hdfDeepCopy(self, ifname, ofname):
        """Copies the internals of an open HDF5 file object (infile) to a second file, 
        replacing the content with stacked data where necessary..."""
        with h5py.File(ofname, "a") as h5o, h5py.File(
            ifname, "r"
        ) as h5f:  # create file, truncate if exists
            # first copy stacked items, adding attributes from infname
            for nxPath in self.stackItems:
                gObj = h5f.get(nxPath, default=None)
                if gObj is None:
                    # print("Path {} not present in input file {}".format(nxPath, ifname))
                    return

                print("Creating dataset at path {}".format(nxPath))
                if nxPath in self.forceDType.keys():
                    print("forcing dType {}".format(self.forceDType[nxPath]))
                    oObj = h5o.create_dataset(
                        nxPath,
                        data=np.stack(self.bDict[nxPath]),
                        dtype=self.forceDType[nxPath],
                        compression="gzip",
                    )
                else:
                    oObj = h5o.create_dataset(
                        nxPath, data=np.stack(self.bDict[nxPath]), compression="gzip"
                    )
                oObj.attrs.update(gObj.attrs)

            # now copy the rest, skipping what is already there.
            for nxPath in self.allItems:
                # do not copy groups...
                oObj = h5o.get(nxPath, default="NonExistentGroup")
                if isinstance(oObj, h5py.Group):
                    # skip copying the group, but ensure all attributes are there..
                    gObj = h5f.get(nxPath, default=None)
                    if gObj is not None:
                        oObj.attrs.update(gObj.attrs)
                    continue  # skippit

                groupLoc = nxPath.rsplit("/", maxsplit=1)[0]
                if len(groupLoc) > 0:
                    gl = h5o.require_group(groupLoc)
                    # copy group attributes
                    oObj = h5f.get(groupLoc)
                    gl.attrs.update(oObj.attrs)

                    try:
                        h5f.copy(nxPath, gl)
                    except (RuntimeError, ValueError):
                        pass
                        # print("Skipping path {}, already exists...".format(nxPath))
                    except:
                        raise

    def listStructure(self, ifname):
        """ reads all the paths to the items into a list """

        def addName(name):
            self.allItems += [name]

        with h5py.File(ifname, "r") as h5f:
            h5f.visit(addName)
        # print(self.allItems)

def processBeam(bfname, ofnamePrepend, NeXpandVersion, ofnameAddon = 'directbeam'):
    # processes the direct- or sample-transmitted beam profiles for later inclusion in the measurments
    bofname = None
    if not bfname.is_file(): print(f"no direct beam image found for measurement {ofnamePrepend}")
    else: 
        bofname = Path(
        bfname.parents[2], f"{ofnamePrepend}_{ofnameAddon}.nxs"
        )
        if bofname.is_file(): bofname.unlink() # remove if exists
        print(f"  nexifying direct beam image from {bfname} in {bofname}")
        NeXify(bfname, bofname, NeXpandVersion)
    return bofname

def runRow(item):
    # structurizes the files from a logbook entry row
    
    # construct all filenames for this row and check if they are there. 
    setComplete = True
    setPathList = []
    for n in np.arange(item.filenum, item.filenum + item.nrep):
        fstr = "{}_{}".format(item.YMD, n)
        filepath = Path('data', str(item.date.year), item.YMD, fstr, 'im_craw.nxs')
        if not filepath.is_file():
            setComplete = False
        else:
            setPathList += [filepath]
    if setComplete is False:
        print(" * * * *  set at YMD:{}, row: {} incomplete, cannot be processed!".format(item.YMD, item.filenum))
        raise ValueError
        return

    # process each file
    for filename in setPathList:
        # check to see if there is a direct beam file:
        dbfname = Path(filename.parent, "beam_profile/im_craw.nxs")
        dbofname = processBeam(dbfname, filename.parts[-2], NeXpandVersion, ofnameAddon = 'directbeam')

        # check to see if there is a sample-transmitted beam file:
        sbfname = Path(filename.parent, "beam_profile_through_sample/im_craw.nxs")
        sbofname = processBeam(sbfname, filename.parts[-2], NeXpandVersion, ofnameAddon = 'samplebeam')

        # now process the actual file
        ofname = Path(filename.parents[1], "{}_expanded.nxs".format(filename.parts[-2]))
        if ofname.is_file():  # remove if exists, for testing only!
            ofname.unlink()
        NeXify(filename, ofname, NeXpandVersion)
        # add an updated thickness from logbook:
        NXUpdate(ofname, "/entry1/sample/thickness", item.samplethickness)

        if (sbofname is not None) and (dbofname is not None):
            # add the beam information from the analyses
            try:
                analyzeBeam(Path(dbofname), imageType = 'both', samplefile = Path(sbofname))
                beamInfoToSampleMeas(
                    ofname,
                    Path(dbofname),
                    Path(sbofname)
                )
            except:
                pass

        # elif dbofname is not None:
        #     analyzeBeam(Path(dbofname), imageType = 'direct')

        # add the sample-transmitted beam data, and make hdf5 link to data
        # elif sbofname is not None:
            # beam analysis can't be done with only a sample-transmitted beam
            # analyzeBeam(Path(sbofname), imageType = 'sample', directfile=Path(dbofname))


    # -/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
    # part 2: stacks the expanded files into multi-frame images
        
    flist = [
        "data/{year}/{YMD}/{YMD}_{ofn}_expanded.nxs".format(
            year=item.date.year, YMD=item.YMD, ofn=(item.filenum + repetition)
        )
        for repetition in range(item.nrep)
    ]

    # for filename in flist:
        # add the direct beam image (quite late stage, I know...)
        # beamFileList = sorted(
        #     Path().glob(
        #         "data/{year}/{YMD}/{YMD}_*_directbeam.nxs".format(
        #             year=item.date.year, YMD=item.YMD
        #         )
        #     )
        # )
        # beamFileNumbers = [
        #     int(item.stem.split("_")[1]) for item in beamFileList
        # ]
        # # get the last beam image with a number less to or equal than the current measurement start number:
        # indices = np.where((np.array(beamFileNumbers) <= item.filenum))[0]
        # if len(indices) != 0:
        #     lastValidIndex = indices[-1]
        #     addBeamFile = beamFileList[lastValidIndex]
        #     print(
        #         "adding direct beam image from file {} to file {} with filenum {}".format(
        #             addBeamFile, filename, item.filenum
        #         )
        #     )
        #     with h5py.File(addBeamFile, "r") as h5b, h5py.File(filename, "a") as h5f:
        #         h5f["/entry1/instrument/detector00/direct_beam_image"][
        #             ...
        #         ] = h5b["/entry1/instrument/detector00/data"][()]

    ofname = flist[0][:-4] + "_stacked.nxs"
    print(flist)
    NXC = NXConcat(flist, ofname, remove=True)
    # NXC.

    # update the sample x value with the data from the logbook
    NXU = NXUpdate(
        fname=ofname,
        NeXusPath="/entry1/sample/transformations/sample_x",
        updateValue=item.positionx,
    )
    NXU = NXUpdate(
        fname=ofname,
        NeXusPath="/entry1/sample/sampleowner",
        updateValue=item.sampleowner,
    )
    NXU = NXUpdate(
        fname=ofname,
        NeXusPath="/entry1/sample/sampleholder",
        updateValue=item.sampleholder,
    )
    NXU = NXUpdate(
        fname=ofname,
        NeXusPath="/entry1/instrument/configuration",
        updateValue=item.configuration,
    )
    NXU = NXUpdate(
        fname=ofname,
        NeXusPath="/entry1/proposal/number",
        updateValue=item.proposal,
    )
    NXU = NXUpdate(
        fname=ofname, NeXusPath="/entry1/user", updateValue=item.user
    )
    # normal (instrument or instrument+cell) background:
    bgYMD = "{year}{month:02d}{day:02d}".format(
        year=item.bgdate.year, month=item.bgdate.month, day=item.bgdate.day
    )
    bgfname = "../../{year}/{bgYMD}/{bgYMD}_{bgn}_expanded_stacked.nxs".format(
        year=item.bgdate.year, bgYMD=bgYMD, bgn=item.bgnumber
    )
    # bgfname = "Y:\Measurements\SAXS002\data\{year}\{bgYMD}\{bgYMD}_{bgn}_expanded_stacked.nxs".format(
    #     year=item.bgdate.year, bgYMD=bgYMD, bgn=item.bgnumber
    # )
    NXU = NXUpdate(
        fname=ofname,
        NeXusPath="/entry1/backgroundfilename",
        updateValue=bgfname,
    )

    # second background which is for a more complete background subtraction where the dispersant and matrix are separated..
    if (item.dbgnumber != -1) and (item.dbgdate.year != 1969): 
        dbgYMD = "{year}{month:02d}{day:02d}".format(
            year=item.dbgdate.year, month=item.dbgdate.month, day=item.dbgdate.day
        )
        dbgfname = "../../{year}/{bgYMD}/{bgYMD}_{bgn}_expanded_stacked.nxs".format(
            year=item.dbgdate.year, bgYMD=dbgYMD, bgn=item.dbgnumber
        )
        # dbgfname = "Y:\Measurements\SAXS002\data\{year}\{bgYMD}\{bgYMD}_{bgn}_expanded_stacked.nxs".format(
        #     year=item.dbgdate.year, bgYMD=dbgYMD, bgn=item.dbgnumber
        # )
        NXU = NXUpdate(
            fname=ofname,
            NeXusPath="/entry1/dispersantbackgroundfilename",
            updateValue=dbgfname,
        )
    NXU = NXUpdate(
            fname=ofname,
            NeXusPath="/entry1/sample/mu",
            updateValue=item.mu,
        )
    NXU = NXUpdate(
            fname=ofname,
            NeXusPath="/entry1/processingpipeline",
            updateValue=item.procpipeline,
        )
    maskYMD = "{year}{month:02d}{day:02d}".format(
        year=item.maskdate.year,
        month=item.maskdate.month,
        day=item.maskdate.day,
    )
    maskfname = "../../Masks/{maskYMD}_{bgn}.nxs".format(
        maskYMD=maskYMD, bgn=item.configuration
    )
    # maskfname = "Y:\Measurements\SAXS002\data\Masks\{maskYMD}_{bgn}.nxs".format(
    #     maskYMD=maskYMD, bgn=item.configuration
    # )
    NXU = NXUpdate(
        fname=ofname, NeXusPath="/entry1/maskfilename", updateValue=maskfname
    )

# Run this magic line to create the output script..
if __name__ == "__main__":
    adict = argparser()
    adict = vars(adict)

    myMeasurementsFile = Path(adict["inputFile"])

    baseMonth = adict["baseDate"]
    year = baseMonth[0:4]

    # do the nexus conversion of every individual file, to a one-frame NeXus image.
    logInstance = readLog(myMeasurementsFile)
    if adict['filenum']==-1: adict['filenum']=None
    lbEntry = logInstance.getMainEntry(YMD = baseMonth, filenum = adict['filenum']) # filenum is a possible addition
    print(f'Running Structurize on {multiprocessing.cpu_count()} cores') 
    Pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    # Pool = multiprocessing.Pool(processes = 1)
    mapParam = [item for index, item in lbEntry.iterrows()]
    rawData = Pool.map(runRow, mapParam)    
    Pool.close()
    Pool.join()
    # for every logbook entry:
    # for index, item in lbEntry.iterrows():
    #     runRow(item)