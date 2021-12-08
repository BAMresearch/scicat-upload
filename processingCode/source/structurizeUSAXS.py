# coding: utf-8

# author: Brian R. Pauw, Glen J. Smales
# date: 2021.08.10

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
import logging
from nexusformat import nexus as nx # once more into the fray
from nexconcat import newConcat
NeXpandVersion = "4.01" # complete rewrites for all but the smallest things. 

# stolen from scicatbam:
def h5Get(filename, h5path, default = 'none', leaveAsArray = False):
    """ get a single value from an HDF5 file, with some error checking and default handling"""
    with h5py.File(filename, "r") as h5f:
        try:
            val = h5f.get(h5path)[()]
            # logging.info('type val {} at key {}: {}'.format(val, h5path, type(val)))
            if isinstance(val, np.ndarray) and (not leaveAsArray):
                if val.size == 1:
                    val = np.array([val.squeeze()])[0]
                else:
                    val = val.mean()
            if isinstance(val, np.int32):
                val = int(val)  # this could go wrong...
            if isinstance(val, np.float32):
                val = float(val)
        except TypeError:
            logging.warning("cannot get value from file path {}, setting to default".format(h5path))
            val = default
    return val

def h5GetDict(filename, keyPaths):
    """creates a dictionary with results extracted from an HDF5 file"""
    resultDict = {}
    for key, h5path in keyPaths.items():
        resultDict[key] = h5Get(filename, key, default=keyPaths[key])
    return resultDict

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
        "-l",
        "--logbookFile",
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

# set up script to add structure to the existing MOUSE files.
class Nex(object):
    # let's try redesigning this one from a different perspective, not doing the stacking, but certainly replacing the original structurize
    core=None
    eigerFile=None
    xenocsFile=None
    outputFile=None
    logbookItem=None
    _requiredInputKeys = ['eigerFile', 'xenocsFile', 'outputFile']

    def __init__(self, eigerFile:Path=None, xenocsFile:Path=None, logbookItem:dict=None, outputFile:Path=None):

        assert isinstance(eigerFile, Path), 'Input Eiger filename must be a Path instance'
        assert isinstance(xenocsFile, Path), 'Input xenocs filename must be a Path instance'
        assert isinstance(logbookItem, pandas.Series), 'Input logbookitem filename must be a dict'
        assert eigerFile.exists(), 'Eiger input file must exist'
        assert xenocsFile.exists(), 'Xenocs input file must exist'
        # assert logbookItem.exists(), 'logbook file must exist'
        self.eigerFile = eigerFile
        self.xenocsFile = xenocsFile
        self.logbookItem = logbookItem
        self.outputFile = outputFile
        if self.outputFile.exists(): self.outputFile.unlink()

        # make sure the root structures exist
        self.createRootStructure()
        self.addInstrument()
        self.addSource()
        self.addDetector()
        self.addUSAXSData()
        self.addCollimation()
        self.addSample()
        self.addBeamstop()
        self.addUser()
        self.addProposal()
        self.addExperimental()
        self.addLinks()

        # if outputFile is not None:
        #     self.save()
        # else:
        #     # for debugging
        #     print(self.core.tree)
        # check inputs
        self.close()
    
    def save(self, ifExists='truncate'):
        ofname = self.outputFile
        """saves the constructed structure to an output nexus file. File will be overwritten if exists"""
        if ifExists=='truncate' or ifExists=='delete':
            self.core.save(filename=ofname)#,'w') # 'w' takes care of truncation
        else:
            logging.error('any other option than file replacement not implemented yet')
    
    def close(self):
        self.core.close()

    def addLinks(self)->None:
        """ adds links to internal values across segments"""
        self.core.entry.instrument.chamber_pressure = nx.NXlinkfield(self.core.entry.sample.chamber_pressure)
        # self.core.entry.instrument.detector.data.rawFrames = nx.NXlinkgroup(target=self.core.entry.data)

    def addUSAXSData(self) -> None:
        usaxsMotors = h5GetDict(
            filename=self.xenocsFile, 
            keyPaths={
                '/saxs/Saxslab/ysam': 0.,    # upstream horizontal translation
                '/saxs/Saxslab/zsam': 0.,    # downstream horizontal translation
                '/saxs/Saxslab/urot': 0.,    # upstream coarse yaw rotation in degrees
                '/saxs/Saxslab/drot': 0.,    # downstream coarse yaw rotation in degrees
                '/saxs/Saxslab/fineyaw': 0., # downstream fine yaw rotation in mrad            
            }
        )
        # # upstream USAXS crystal
        self.core.entry.instrument.upstream_crystal=nx.NXcrystal(
            nx.NXtransformations(
                nx.NXfield(name='y_translation', value = np.float64(usaxsMotors['/saxs/Saxslab/ysam']), attrs={
                    'depends_on':'.', 
                    'offset': [0.,0.,0.],
                    'offset_units':'mm',
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[1, 0.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
                nx.NXfield(name='yaw', value = np.float64(usaxsMotors['/saxs/Saxslab/urot']), attrs={
                    'depends_on':'./y_translation', 
                    'units':'deg',
                    'transformation_type':'rotation',
                    'vector':[0.0, 1.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
            ),
            nx.NXfield(name='usage', value='Bragg'),
            nx.NXfield(name='type', value='Si'),
            nx.NXfield(name='order_no', value=1, attrs={'units':''}),
            nx.NXfield(name='d_spacing', value=1.920155716e-10, attrs={'units':'m'}),
            nx.NXfield(name='miller_indices', value=220, attrs={'units':''}),
            nx.NXnote(name='note', value='upstream four-bounce Si(220) channel-cut crystal used as Bonse-Hart collimation/monochromator crystal. Rotation axis vertical around first reflection.', dtype='U')
            )
        # # downstream USAXS crystal
        self.core.entry.instrument.downstream_crystal=nx.NXcrystal(
            nx.NXtransformations(
                nx.NXfield(name='y_translation', value = np.float64(usaxsMotors['/saxs/Saxslab/zsam']), attrs={
                    'depends_on':'.', 
                    'offset': [0.,0.,0.],
                    'offset_units':'mm',
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[1, 0.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
                nx.NXfield(name='fineyaw', value = np.float64(usaxsMotors['/saxs/Saxslab/fineyaw']), attrs={
                    'depends_on':'./y_translation', 
                    'units':'mrad',
                    'transformation_type':'rotation',
                    'vector':[0.0, 1.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
                nx.NXfield(name='yaw', value = np.float64(usaxsMotors['/saxs/Saxslab/drot']), attrs={
                    'depends_on':'./fineyaw', 
                    'units':'deg',
                    'transformation_type':'rotation',
                    'vector':[0.0, 1.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
            ),
            nx.NXfield(name='usage', value='Bragg'),
            nx.NXfield(name='type', value='Si'),
            nx.NXfield(name='order_no', value=1, attrs={'units':''}),
            nx.NXfield(name='d_spacing', value=1.920155716e-10, attrs={'units':'m'}),
            nx.NXfield(name='miller_indices', value=220, attrs={'units':''}),
            nx.NXnote(name='note', value='downstream four-bounce Si(220) channel-cut crystal used as Bonse-Hart analyzer crystal. Rotation axis vertical around first reflection.', dtype='U')
            )
        # # main data

    def addDetector(self):

        detectorExtract = h5GetDict(
            filename=self.xenocsFile, 
            keyPaths={
                '/saxs/Saxslab/detx': 0, # mm
                '/saxs/Saxslab/dety': 0, # mm
                '/saxs/Saxslab/detz': 0, # mm
                '/saxs/Saxslab/detx_enc': 0, # encoder readout
                '/saxs/Saxslab/detector_dist': 0, # detector "distance"
            }
        )
        # '../../../Measurements/SAXS002/data/2021/20210805/20210805_2/USAXS_4/eiger_0097366_master.h5'
        inData = nx.nxload(self.eigerFile, 'r')
        # # detector information
        self.core.entry.instrument.detector00= inData['/entry/instrument/detector'].copy(expand_external=True)
        self.core.entry.instrument.detector00.attrs.update({'default':'data'})
        self.core.entry.instrument.detector00.frames = inData['/entry/data'].copy(expand_external=True)
        # # image frames are stored here
        self.core.entry.instrument.detector00.data=nx.NXdata(attrs={'default':'average'})
        # TODO: needs updating in case we have more than.. 100 frames, I think?
        inDat=inData['entry/data/data_000001'][()].astype(np.float64)
        self.core.entry.instrument.detector00.data.average=nx.NXfield(
            value=inDat.mean(axis=0), 
            compression='gzip',
            attrs={'uncertainties': 'uncertainties_poisson', 'units':'counts'}
            )
        self.core.entry.instrument.detector00.data.uncertainties_poisson=nx.NXfield(
            value=np.sqrt(inDat.mean(axis=0)).clip(1), 
            compression='gzip',
            attrs={'units': 'counts'}
            )
        self.core.entry.instrument.detector00.data.uncertainties_sem=nx.NXfield(
            value=inDat.std(axis=0) / np.sqrt(inDat.shape[0]), 
            compression='gzip',
            attrs={'units': 'counts', 'note': 'standard error on the mean (ddof=1)'}
            )
        # self.core.entry.instrument.detector.data.frame_exposure_times=nx.NXfield(np.array(USAXSDict['countTimes']))
        # hope this still does something useful. 
        inData.close()
        # now let's move this detector into the right place:
        self.core.entry.instrument.detector00.transformations = nx.NXtransformations(
            nx.NXfield(name='det_x', value = np.float64(detectorExtract['/saxs/Saxslab/detx']), attrs={
                'depends_on':'.', 
                'offset': [0.,0.,0.],
                'offset_units':'mm',
                'units':'mm',
                'transformation_type':'translation',
                'vector':[0.0, 0.0, 1.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                }),
            nx.NXfield(name='det_y', value = np.float64(detectorExtract['/saxs/Saxslab/dety']), attrs={
                'depends_on':'./det_z', 
                'offset': [0.,0.,0.],
                'offset_units':'mm',
                'units':'mm',
                'transformation_type':'translation',
                'vector':[1.0, 0.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                }),
            nx.NXfield(name='det_z', value = np.float64(detectorExtract['/saxs/Saxslab/detz']), attrs={
                'depends_on':'./det_x', 
                'offset': [0.,0.,0.],
                'offset_units':'mm',
                'units':'mm',
                'transformation_type':'translation',
                'vector':[0.0, 1.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                }),
            nx.NXfield(name='euler_a', value = 0., attrs={
                'depends_on':'./det_y', 
                'offset': [0.,0.,0.],
                'offset_units':'mm',
                'units':'deg',
                'transformation_type':'rotation',
                'vector':[0.0, 0.0, 1.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                }),
            nx.NXfield(name='euler_b', value = 0., attrs={
                'depends_on':'./euler_a', 
                'offset': [0.,0.,0.],
                'offset_units':'mm',
                'units':'deg',
                'transformation_type':'rotation',
                'vector':[0.0, 0.0, 1.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                }),
            nx.NXfield(name='euler_c', value = 0., attrs={
                'depends_on':'./euler_b', 
                'offset': [0.,0.,0.],
                'offset_units':'mm',
                'units':'deg',
                'transformation_type':'rotation',
                'vector':[0.0, 0.0, 1.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                }),
            nx.NXfield(name='type', value='pixel'),
            nx.NXfield(name='virtual_pixel_correction_applied', value=1),
        )
        self.core.entry.instrument.detector00.det_x_encoder=nx.NXfield(value = np.float64(detectorExtract['/saxs/Saxslab/detx_enc']), attrs={
                'depends_on':'.', 
                'offset': [0.,0.,0.],
                'offset_units':'mm',
                'units':'mm',
                'transformation_type':'translation',
                'vector':[0.0, 0.0, 1.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                'note':'encoder readout module mounted to the side of the main detector translation stage',
                }),

    def addSource(self):
        # source information
        sourceExtract = h5GetDict(
            filename=self.xenocsFile, 
            keyPaths={
                '/saxs/Saxslab/source_kV': 0, # in kV, current source only.
                '/saxs/Saxslab/source_ma': 0, # in mA
                '/saxs/Saxslab/source_type': 0, # 'Genix3D or whatever'
                '/saxs/Saxslab/dual': 0, # 'in.... mm I think'
            }
        )
        self.core.entry.instrument.source=nx.NXsource(
            nx.NXtransformations(
                nx.NXfield(name='dual_y', value = np.float64(sourceExtract['/saxs/Saxslab/dual']), attrs={
                    'depends_on':'.', 
                    'offset': [0.,0.,-1600.],
                    'offset_units':'mm',
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[1., 0.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
            ),
            nx.NXfield(name='current', value = np.float64(sourceExtract['/saxs/Saxslab/source_ma']), attrs={'units': 'mA'}, dtype=np.float64),
            nx.NXfield(name='voltage', value = np.float64(sourceExtract['/saxs/Saxslab/source_kV']), attrs={'units': 'kV'}, dtype=np.float64),
            nx.NXnote(name='name', value=sourceExtract['/saxs/Saxslab/source_type'], dtype='U'),
            nx.NXnote(name='probe', value='X-ray', dtype='U'),
            nx.NXnote(name='type', value='Fixed Tube X-ray', dtype='U'),
        )

    def addBeamstop(self):
        beamstopExtract = h5GetDict(
            filename=self.xenocsFile, 
            keyPaths={
                '/saxs/Saxslab/bsr': 0, # rotation in.. maybe degrees
                '/saxs/Saxslab/bsz': 0, # z movement in mm
                '/saxs/Saxslab/bsh': 0, # pseudomotor I think
                '/saxs/Saxslab/bsv': 0, # same but vertical
                '/saxs/Saxslab/bstop': 0, # llength of bstop arm in mm, maybe?
            }
        )
        # # beamstop
        self.core.entry.instrument.beamstop=nx.NXbeam_stop(
            nx.NXtransformations(
                nx.NXfield(name='bsz', value = np.float64(beamstopExtract['/saxs/Saxslab/bsz']), attrs={
                    'depends_on':'/entry/detector/transformations/det_x', 
                    'offset': [0.,0.,-10.],
                    'offset_units':'mm',
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[0., 1.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                 }),
                nx.NXfield(name='bsr', value = np.float64(beamstopExtract['/saxs/Saxslab/bsr']), attrs={
                    'depends_on':'./bsz', 
                    'offset': [0.,0.,-10.],
                    'offset_units':'mm',
                    'units':'deg',
                    'transformation_type':'rotation',
                    'vector':[0., 0.0, 1.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                }),
                nx.NXfield(name='arm_length', value = np.float64(beamstopExtract['/saxs/Saxslab/bstop']), attrs={
                    'depends_on':'./bsz', 
                    'offset': [0.,0.,-10.],
                    'offset_units':'mm',
                    'units':'mm',
                    'transformation_type':'rotation',
                    'vector':[0., 0.0, 1.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                }),
            ),
            nx.NXfield(name='beamstop_in', value=self.logbookItem.beamstop, dtype=bool, attrs={'note':'Is the beamstop in front of the beam?'})
            )

    def addCollimation(self):
        collimationExtract = h5GetDict(
            filename=self.xenocsFile, 
            keyPaths={
                '/saxs/Saxslab/hg1': 0, # horizontal gap 1
                '/saxs/Saxslab/hg2': 0, # horizontal gap 2
                '/saxs/Saxslab/hg3': 0, # horizontal gap 3
                '/saxs/Saxslab/hp1': 0, # horizontal position 1
                '/saxs/Saxslab/hp2': 0, # horizontal position 2
                '/saxs/Saxslab/hp3': 0, # horizontal position 3
                '/saxs/Saxslab/vg1': 0, # vertical gap 1
                '/saxs/Saxslab/vg2': 0, # vertical gap 2
                '/saxs/Saxslab/vg3': 0, # vertical gap 3
                '/saxs/Saxslab/vp1': 0, # vertical position 1
                '/saxs/Saxslab/vp2': 0, # vertical position 2
                '/saxs/Saxslab/vp3': 0, # vertical position 3
                '/saxs/Saxslab/s1bot': 0, # slit 1 bottom blade position
                '/saxs/Saxslab/s1top': 0, # slit 1 top blade position
                '/saxs/Saxslab/s1hl': 0, # slit 1 left blade position
                '/saxs/Saxslab/s1hr': 0, # slit 1 right blade position
                '/saxs/Saxslab/s2bot': 0, # slit 2 bottom blade position
                '/saxs/Saxslab/s2top': 0, # slit 2 top blade position
                '/saxs/Saxslab/s2hl': 0, # slit 2 left blade position
                '/saxs/Saxslab/s2hr': 0, # slit 2 right blade position
                '/saxs/Saxslab/s3bot': 0, # slit 3 bottom blade position
                '/saxs/Saxslab/s3top': 0, # slit 3 top blade position
                '/saxs/Saxslab/s3hl': 0, # slit 3 left blade position
                '/saxs/Saxslab/s3hr': 0, # slit 3 right blade position
                '/saxs/Saxslab/saxsconf_l1': 0, # distance between slit 1 and 2
                '/saxs/Saxslab/saxsconf_l2': 0, # distance between slit 2 and 3
                '/saxs/Saxslab/saxsconf_l3': 0, # distance between slit 3 and sample
                
                # start_timestamp # not in a proper format
                # wavelength (maybe)
            }
        )
        # # collimation section:
        self.core.entry.instrument.collimator1=nx.NXcollimator(
            nx.NXtransformations(
                nx.NXfield(name='cen_y', value = np.float64(collimationExtract['/saxs/Saxslab/hp1']), attrs={
                    'depends_on':'.', 
                    'offset': [0.,0.,-1*(
                        np.float64(collimationExtract['/saxs/Saxslab/saxsconf_l3'])+
                        np.float64(collimationExtract['/saxs/Saxslab/saxsconf_l2'])+
                        np.float64(collimationExtract['/saxs/Saxslab/saxsconf_l1'])
                        )],
                    'offset_units':'mm',
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[1., 0.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
                nx.NXfield(name='cen_z', value = np.float64(collimationExtract['/saxs/Saxslab/vp1']), attrs={
                    'depends_on':'./cen_y', 
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[0., 1.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
            ),
            nx.NXaperture(name="h_aperture", value = np.float64(collimationExtract['/saxs/Saxslab/hg1']), attrs={'units':'mm'}),
            nx.NXaperture(name="v_aperture", value = np.float64(collimationExtract['/saxs/Saxslab/vg1']), attrs={'units':'mm'}),
            nx.NXfield(name='material', value='Ge'),
            nx.NXnote(name='note', value='upstream scatterless, four-blade slit', dtype='U'),
            nx.NXfield(name="blade_bottom", value = np.float64(collimationExtract['/saxs/Saxslab/s1bot']), attrs={'units':'mm'}),
            nx.NXfield(name="blade_top", value = np.float64(collimationExtract['/saxs/Saxslab/s1top']), attrs={'units':'mm'}),
            nx.NXfield(name="blade_negy", value = np.float64(collimationExtract['/saxs/Saxslab/s1hl']), attrs={'units':'mm'}),
            nx.NXfield(name="blade_posy", value = np.float64(collimationExtract['/saxs/Saxslab/s1hr']), attrs={'units':'mm'}),
        )
        self.core.entry.instrument.collimator2=nx.NXcollimator(
            nx.NXtransformations(
                nx.NXfield(name='cen_y', value = np.float64(collimationExtract['/saxs/Saxslab/hp2']), attrs={
                    'depends_on':'.', 
                    'offset': [0.,0.,-1*(
                        np.float64(collimationExtract['/saxs/Saxslab/saxsconf_l3'])+
                        np.float64(collimationExtract['/saxs/Saxslab/saxsconf_l2'])
                        )],
                    'offset_units':'mm',
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[1., 0.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
                nx.NXfield(name='cen_z', value = np.float64(collimationExtract['/saxs/Saxslab/vp2']), attrs={
                    'depends_on':'./cen_y', 
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[0., 1.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
            ),
            nx.NXaperture(name="h_aperture", value = np.float64(collimationExtract['/saxs/Saxslab/hg2']), attrs={'units':'mm'}),
            nx.NXaperture(name="v_aperture", value = np.float64(collimationExtract['/saxs/Saxslab/vg2']), attrs={'units':'mm'}),
            nx.NXfield(name='material', value='Ge'),
            nx.NXnote(name='note', value='middle scatterless, four-blade slit', dtype='U'),
            nx.NXfield(name="blade_bottom", value = np.float64(collimationExtract['/saxs/Saxslab/s2bot']), attrs={'units':'mm'}),
            nx.NXfield(name="blade_top", value = np.float64(collimationExtract['/saxs/Saxslab/s2top']), attrs={'units':'mm'}),
            nx.NXfield(name="blade_negy", value = np.float64(collimationExtract['/saxs/Saxslab/s2hl']), attrs={'units':'mm'}),
            nx.NXfield(name="blade_posy", value = np.float64(collimationExtract['/saxs/Saxslab/s2hr']), attrs={'units':'mm'}),        )
        self.core.entry.instrument.collimator3=nx.NXcollimator(
            nx.NXtransformations(
                nx.NXfield(name='cen_y', value = np.float64(collimationExtract['/saxs/Saxslab/hp3']), attrs={
                    'depends_on':'.', 
                    'offset': [0.,0.,-1*np.float64(collimationExtract['/saxs/Saxslab/saxsconf_l3'])],
                    'offset_units':'mm',
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[1., 0.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
                nx.NXfield(name='cen_z', value = np.float64(collimationExtract['/saxs/Saxslab/vp3']), attrs={
                    'depends_on':'./cen_y', 
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[0., 1.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
            ),
            nx.NXaperture(name="h_aperture", value = np.float64(collimationExtract['/saxs/Saxslab/hg3']), attrs={'units':'mm'}),
            nx.NXaperture(name="v_aperture", value = np.float64(collimationExtract['/saxs/Saxslab/vg3']), attrs={'units':'mm'}),
            nx.NXfield(name='material', value='Si'),
            nx.NXnote(name='note', value='downstream scatterless, four-blade slit', dtype='U'),
            nx.NXfield(name="blade_bottom", value = np.float64(collimationExtract['/saxs/Saxslab/s3bot']), attrs={'units':'mm'}),
            nx.NXfield(name="blade_top", value = np.float64(collimationExtract['/saxs/Saxslab/s3top']), attrs={'units':'mm'}),
            nx.NXfield(name="blade_negy", value = np.float64(collimationExtract['/saxs/Saxslab/s3hl']), attrs={'units':'mm'}),
            nx.NXfield(name="blade_posy", value = np.float64(collimationExtract['/saxs/Saxslab/s3hr']), attrs={'units':'mm'}),        
        )

    def addSample(self):
        sampleExtract = h5GetDict(
            filename=self.xenocsFile, 
            keyPaths={
                '/saxs/Saxslab/ysam': 0.,    # sample horizontal translation if not usaxs
                '/saxs/Saxslab/zsam': 0.,    # sample vertical translation if not usaxs
                '/saxs/Saxslab/yheavy': 0.,    # sample horizontal translation on heavy stage
                '/saxs/Saxslab/zheavy': 0.,    # sample vertical translation on heavy stage
                '/saxs/Saxslab/chamber_pressure': 0., # chamber pressure in mbar
                '/saxs/Saxslab/sample_ID': '0', # sample ID
                '/saxs/Saxslab/Meas.Description': '0', # sample ID
                '/saxs/Saxslab/sample_thickness': 0, # in centimeters, if I'm not mistaken
                '/saxs/Saxslab/wavelength': 0, # in ongstroam
            }
        )
        # sample section
        if self.logbookItem.usaxs: 
            y=np.float64(sampleExtract['/saxs/Saxslab/yheavy'])
            z=np.float64(sampleExtract['/saxs/Saxslab/zheavy'])
        else:
            y=np.float64(sampleExtract['/saxs/Saxslab/ysam'])
            z=np.float64(sampleExtract['/saxs/Saxslab/zsam'])

        self.core.entry.sample=nx.NXsample(
            nx.NXtransformations(
                nx.NXfield(name='sample_x', value = self.logbookItem.positionx, attrs={
                    'depends_on':'.', 
                    'offset': [0.,0.,0.],
                    'offset_units':'mm',
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[0., 0.0, 1.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
                nx.NXfield(name='sample_y', value = y, attrs={
                    'depends_on':'./sample_x', 
                    'offset': [0.,0.,0.],
                    'offset_units':'mm',
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[1., 0.0, 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
                nx.NXfield(name='sample_z', value = z, attrs={
                    'depends_on':'./sample_y', 
                    'offset': [0.,0.,0.],
                    'offset_units':'mm',
                    'units':'mm',
                    'transformation_type':'translation',
                    'vector':[0., 1., 0.0], # Diamond units: from our units y=[1,0,0]; z=[0,1,0]; x=[0,0,1]
                    }),
            ), 
            nx.NXbeam(
                nx.NXfield(name='incident_wavelength', value = np.float64(sampleExtract['/saxs/Saxslab/wavelength']), attrs={'units': 'angstrom'}),
                nx.NXfield(name='flux', value = 1., attrs={'units': 's-1'}) # to be filled in later
            ), 
            nx.NXfield(name='mu', value = self.logbookItem.mu, attrs={'units': 'm-1', 'note': 'sample-phase overall X-ray absorption coefficient at incident energy'}, dtype=np.float64), # to be filled in later
            nx.NXfield(name='thickness', value = self.logbookItem.samplethickness, attrs={'units': 'm'}, dtype=np.float64), # to be filled in later
            nx.NXfield(name='transmission', value = 1., attrs={'units': '', 'note': 'overall X-ray transmission factor'}, dtype=np.float64), 
            nx.NXfield(name='name', value = sampleExtract['/saxs/Saxslab/Meas.Description'], dtype='U'),
            nx.NXfield(name='sample_holder', value = self.logbookItem.sampleholder),
            nx.NXfield(name='sample_owner', value = self.logbookItem.sampleowner),
            nx.NXnote(name='sample_id', value=self.logbookItem.sampleid, dtype='U'),
            nx.NXnote(name='sample_name', value=self.logbookItem.samplename, dtype='U'),
            nx.NXnote(name='sample_notes', value=self.logbookItem.notes, dtype='U'),
            nx.NXfield(name='chamber_pressure', value = np.float64(sampleExtract['/saxs/Saxslab/chamber_pressure']), attrs={'units': 'hPa'}, dtype=np.float64),
        )
        self.core.entry.sample.attrs.update({'depends_on': './transformations/sample_y'})

        # # sample container and sample make-up. This will need to be generated or filled in later once the database is up and running again. 
        self.core.entry.sample.sample_container = nx.NXgroup(attrs={"NX_class":"NXcontainer"})
        self.core.entry.sample.sample_container.upstream_window = nx.NXgroup(
            nx.NXfield(name='atomic_composition', value='CH2', dtype='U'),
            nx.NXfield(name='density', value=2.5, attrs={'units': 'g/cc'}, dtype=np.float64),
            nx.NXfield(name='thickness', value=0.01, attrs={'units': 'mm'}, dtype=np.float64),
            nx.NXfield(name='geometry', value='planar'),
            nx.NXnote(name='description', value='Scotch Magic Tape', dtype='U'),
            attrs={"NX_class":"NXmaterial"},
        )
        self.core.entry.sample.component1=nx.NXsample_component(
            nx.NXfield(name='atomic_composition', value='SiO2'),
            nx.NXfield(name='density', value=2.2, attrs={'units': 'g/cc'}, dtype=np.float64),
            nx.NXfield(name='volume_fraction', value=0.68, attrs={'units': ''}, dtype=np.float64),
            nx.NXfield(name='geometry', value='spherical'),
            nx.NXnote(name='description', value='500 nm silica particles', dtype='U'),
            # keep nesting if required:
            # nx.NXlinkgroup(target=core.entry.sample.component1),
            attrs={"NX_class":"NXmaterial"},
        )
        self.core.entry.sample.component0=nx.NXsample_component(
            nx.NXfield(name='atomic_composition', value='O0.79N0.21'),
            nx.NXfield(name='density', value=0.0001, attrs={'units': 'g/cc'}, dtype=np.float64),
            nx.NXfield(name='thickness', value=0.1, attrs={'units': 'mm'}, dtype=np.float64),
            nx.NXfield(name='volume_fraction', value=0.32, attrs={'units': ''}, dtype=np.float64),
            nx.NXfield(name='geometry', value='planar'),
            nx.NXnote(name='description', value='air', dtype='U'),
            nx.NXlinkgroup(name='nested_component', target=self.core.entry.sample.component1),
            attrs={"NX_class":"NXmaterial"},
        )
        self.core.entry.sample.sample_container.nested_component=nx.NXlinkgroup(target=self.core.entry.sample.component0)
        self.core.entry.sample.sample_container.downstream_window = nx.NXlinkgroup(target=self.core.entry.sample.sample_container.upstream_window)

    def addProposal(self)-> None:
        self.core.entry.proposal=nx.NXnote(
            nx.NXfield(name='proposal_id', value=self.logbookItem.proposal),
            nx.NXfield(name='proposal_user', value=self.logbookItem.sampleowner),
            dtype='U'
            ) 
    
    def addUser(self)->None:
        self.core.entry.user=nx.NXuser(value=self.logbookItem.user, dtype='U') # updated later

    def addInstrument(self)->None:
        # instrument description
        self.core.entry.instrument=nx.NXinstrument(attrs={'default': 'detector'})
        # # multilayer monochromator information 
        self.core.entry.instrument.monochromator=nx.NXmonochromator(
            nx.NXtransformations(), 
            nx.NXmirror(
                nx.NXfield(name='type', value='multi', dtype='U'),
                nx.NXfield(name='description', value='Xenocs Fox3D mirror', dtype='U'),
                nx.NXfield(name='interior_atmosphere', value='vacuum', dtype='U')
            ),
            )
        self.core.entry.instrument.configuration=nx.NXfield(value=self.logbookItem.configuration, attrs={'note': 'instrument configuration preset'})
        self.core.entry.instrument.name=nx.NXnote(value='The MOUSE', dtype='U')
        self.core.entry.instrument.references=nx.NXgroup(
            nx.NXcite(name='instrument_paper',
            author='Glen J. Smales and Brian R. Pauw',
            description = 'The main instrument paper for the MOUSE instrument',
            url='https://doi.org/10.1088/1748-0221/16/06/P06034',
            doi='10.1088/1748-0221/16/06/P06034',
            bibtex="""
                @article{Smales_2021,
                doi = {10.1088/1748-0221/16/06/p06034},
                url = {https://doi.org/10.1088/1748-0221/16/06/p06034},
                year = 2021,
                month = {jun},
                publisher = {{IOP} Publishing},
                volume = {16},
                number = {06},
                pages = {P06034},
                author = {Glen J. Smales and Brian R. Pauw},
                title = {The {MOUSE} project: a meticulous approach for obtaining traceable, wide-range X-ray scattering information},
                journal = {Journal of Instrumentation},
                abstract = {Herein, we provide a “systems architecture”-like overview and detailed discussions of the methodological and instrumental components that, together, comprise the “MOUSE” project (Methodology Optimization for Ultrafine Structure Exploration). The MOUSE project provides scattering information on a wide variety of samples, with traceable dimensions for both the scattering vector (q) and the absolute scattering cross-section (I). The measurable scattering vector-range of 0.012≤ q (nm-1) ≤ 92, allows information across a hierarchy of structures with dimensions ranging from ca. 0.1 to 400 nm. In addition to details that comprise the MOUSE project, such as the organisation and traceable aspects, several representative examples are provided to demonstrate its flexibility. These include measurements on alumina membranes, the tobacco mosaic virus, and dual-source information that overcomes fluorescence limitations on ZIF-8 and iron-oxide-containing carbon catalyst materials.}
                }
            """,
            dtype='U'
            ),
            nx.NXcite(name='datacorrections_paper',
            author='Brian Richard Pauw, A. J. Smith, T. Snow, N. J. Terrill and A. F. Thünemann',
            description = 'The data corrections applied to the processed data',
            url='https://doi.org/10.1107/S1600576717015096',
            doi='10.1107/S1600576717015096',
            bibtex="""
                @article{Pauw:vg5075,
                author = "Pauw, B. R. and Smith, A. J. and Snow, T. and Terrill, N. J. and Th{\"{u}}nemann, A. F.",
                title = "{The modular small-angle X-ray scattering data correction sequence}",
                journal = "Journal of Applied Crystallography",
                year = "2017",
                volume = "50",
                number = "6",
                pages = "1800--1811",
                month = "Dec",
                doi = {10.1107/S1600576717015096},
                url = {https://doi.org/10.1107/S1600576717015096},
                abstract = {Data correction is probably the least favourite activity amongst users experimenting with small-angle X-ray scattering: if it is not done sufficiently well, this may become evident only during the data analysis stage, necessitating the repetition of the data corrections from scratch. A recommended comprehensive sequence of elementary data correction steps is presented here to alleviate the difficulties associated with data correction, both in the laboratory and at the synchrotron. When applied in the proposed order to the raw signals, the resulting absolute scattering cross section will provide a high degree of accuracy for a very wide range of samples, with its values accompanied by uncertainty estimates. The method can be applied without modification to any pinhole-collimated instruments with photon-counting direct-detection area detectors.},
                keywords = {small-angle scattering, accuracy, methodology, data correction},
                }
            """,
            dtype='U'
            ),
            nx.NXcite(name='usaxs_paper',
            author='Brian R. Pauw, Andrew J. Smith, Tim Snow, Olga Shebanova, John P. Sutter, Jan Ilavsky, Daniel Hermida-Merino, Glen J. Smales, Nicholas J. Terrill, Andreas F. Thuenemann and Wim Bras',
            description = 'The publication on the USAXS module',
            url='https://doi.org/10.1107/S1600577521003313',
            doi='10.1107/S1600577521003313',
            bibtex="""
               @article{Pauw:ok5040,
                author = "Pauw, Brian R. and Smith, Andrew J. and Snow, Tim and Shebanova, Olga and Sutter, John P. and Ilavsky, Jan and Hermida-Merino, Daniel and Smales, Glen J. and Terrill, Nicholas J. and Th{\"{u}}nemann, Andreas F. and Bras, Wim",
                title = "{Extending synchrotron SAXS instrument ranges through addition of a portable, inexpensive USAXS module with vertical rotation axes}",
                journal = "Journal of Synchrotron Radiation",
                year = "2021",
                volume = "28",
                number = "3",
                pages = "824--833",
                month = "May",
                doi = {10.1107/S1600577521003313},
                url = {https://doi.org/10.1107/S1600577521003313},
                abstract = {Ultra-SAXS can enhance the capabilities of existing synchrotron SAXS/WAXS beamlines. A compact ultra-SAXS module has been developed, which extends the measurable {\it q}-range with 0.0015 {$\leq$} {\it q} (nm${\sp {$-$}1}$) {$\leq$} 0.2, allowing structural dimensions in the range 30 {$\leq$} {\it D} (nm) {$\leq$} 4000 to be probed in addition to the range covered by a high-end SAXS/WAXS instrument. By shifting the module components in and out on their respective motor stages, SAXS/WAXS measurements can be easily and rapidly interleaved with USAXS measurements. The use of {\it vertical} crystal rotation axes (horizontal diffraction) greatly simplifies the construction, at minimal cost to efficiency. In this paper, the design considerations, realization and synchrotron findings are presented. Measurements of silica spheres, an alumina membrane, and a porous carbon catalyst are provided as application examples.},
                keywords = {ultra-SAXS, USAXS, X-ray scattering, instrumentation, module},
                }
            """,
            dtype='U'
            ),
        )

    def addExperimental(self)->None: # adds experimental details needed for processing
        # normal (instrument or instrument+cell) background:
        bgYMD = f"{self.logbookItem.bgdate.year}{self.logbookItem.bgdate.month:02d}{self.logbookItem.bgdate.day:02d}"
        bgfname = f"../../{self.logbookItem.bgdate.year}/{bgYMD}/{bgYMD}_{self.logbookItem.bgnumber}_expanded_stacked.nxs"

        maskYMD = f"{self.logbookItem.maskdate.year}{self.logbookItem.maskdate.month:02d}{self.logbookItem.maskdate.day:02d}"
        maskfname = f"../../Masks/{maskYMD}_{self.logbookItem.configuration}.nxs"

        self.core.entry.experimental=nx.NXgroup(
            nx.NXfield(name='background_filename', value=bgfname, dtype='U'),
            nx.NXfield(name='mask_filename', value=maskfname, dtype='U'),
            nx.NXfield(name='processing_pipeline', value=self.logbookItem.procpipeline, dtype='U'),
        )

        # second background which is for a more complete background subtraction where the dispersant and matrix are separated..
        if (item.dbgnumber != -1) and (item.dbgdate.year != 1969): 
            dbgYMD = f"{self.logbookItem.dbgdate.year}{self.logbookItem.dbgdate.month:02d}{self.logbookItem.dbgdate.day:02d}"
            dbgfname = f"../../{self.logbookItem.dbgdate.year}/{bgYMD}/{bgYMD}_{self.logbookItem.dbgnumber}_expanded_stacked.nxs"
            self.core.entry.experimental.dispersant_background_filename = nx.NXfield(value=dbgfname, dtype='U')
        self.core.entry.experimental.is_insitu=nx.NXfield(value=self.logbookItem.insitu, dtype=bool, attrs={'note':'Is this an insitu measurement (reduced beam-and-transmission measurmenets)?'})
        self.core.entry.experimental.is_usaxs=nx.NXfield(value=self.logbookItem.usaxs, dtype=bool, attrs={'note':'Is the instrument configured with USAXS towers in place?'})

    def createRootStructure(self) -> None:
        self.core = nx.nxload(self.outputFile, 'w') # nx.NXroot(attrs={'default': 'entry', 'note': 'trial version of a USAXS data writer for 1D data from the MOUSE 20210610'})
        self.core.attrs.update({'default': 'entry'})
        self.core.entry=nx.NXentry(attrs={'default': 'angles'})

def runRow(item):
    # structurizes the files from a logbook entry row
    
    # construct all filenames for this row and check if they are there. 
    setComplete = True
    setPathList = []
    # this has to be redone, using Path.glob probably, it's ugly AF right now, and probs slow too. 
    if item.usaxs==True:
        for n in np.arange(item.filenum, item.filenum + item.nrep):
            # globber = Path('data', str(item.date.year), item.YMD, f"{item.YMD}_{n}").glob('*/im_craw.nxs')
            # pathList = sorted(globber)
            pathList= [Path('data', str(item.date.year), item.YMD, f"{item.YMD}_{n}", f'USAXS_{j}', 'im_craw.nxs') for j in range(280)]
            for filepath in pathList:
                assert filepath.exists(), logging.warning(f'USAXS file {filepath} in series number {n} is missing')
            setPathList += pathList # this will probably go wrong with more than one repetition.. 
    else:
        for n in np.arange(item.filenum, item.filenum + item.nrep):
            filepath = Path('data', str(item.date.year), item.YMD, f"{item.YMD}_{n}", 'im_craw.nxs')
            if not filepath.is_file():
                setComplete = False
            else:
                setPathList += [filepath]

    if setComplete == False:
        print(" * * * *  set at YMD:{}, row: {} incomplete, cannot be processed!".format(item.YMD, item.filenum))
        raise ValueError
        return

    # process each file
    flist = []
    for filename in setPathList:
        outputFile = Path(filename.parents[1], f'USAXS_{filename.parts[-2].split("_")[-1]}.nxs')
        # skipping for testing:
        Nex(
            xenocsFile = filename, 
            eigerFile = sorted(filename.parent.glob('*master.h5'))[0], 
            logbookItem=item, 
            outputFile = outputFile)
        flist+=[outputFile]

    # -/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
    # part 2: stacks the expanded files into multi-frame images
    if item.usaxs:
        pass
    else:
        flist = [
            Path(f"data/{item.date.year}/{item.YMD}/{item.YMD}_{(item.filenum + repetition)}_expanded.nxs")
            for repetition in range(item.nrep)
        ]

    filename = flist[0]
    ofname = Path(filename.parents[1], f'USAXS_{filename.parts[-2].split("_")[-1]}' + "_stacked.nxs")
    print(flist)
    NXC = newConcat(filenames = flist, outputFile = ofname)

# Run this magic line to create the output script..
if __name__ == "__main__":
    # forgot how this goes again:
    logging.basicConfig(
        filename = Path('structurizeUSAXS_runlog.log'), 
        # encoding = 'utf-8', # <-- from Python 3.9
        filemode = 'w', # <-- if we want a new log every time.
        level = logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S'
        )
        
    adict = argparser()
    adict = vars(adict)

    myMeasurementsFile = Path(adict["logbookFile"])

    baseMonth = adict["baseDate"]
    year = baseMonth[0:4]

    # do the nexus conversion of every individual file, to a one-frame NeXus image.
    logInstance = readLog(myMeasurementsFile)
    if adict['filenum']==-1: adict['filenum']=None
    lbEntry = logInstance.getMainEntry(YMD = baseMonth, filenum = adict['filenum']) # filenum is a possible addition
    # print(f'Running Structurize on {multiprocessing.cpu_count()} cores') 
    # Pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    # # Pool = multiprocessing.Pool(processes = 1)
    # mapParam = [item for index, item in lbEntry.iterrows()]
    # rawData = Pool.map(runRow, mapParam)    
    # Pool.close()
    # Pool.join()

    # for every logbook entry:
    for index, item in lbEntry.iterrows():
        runRow(item)
