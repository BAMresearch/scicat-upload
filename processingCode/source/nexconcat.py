# coding: utf-8

import hdf5plugin
import h5py
from pathlib import Path
import numpy as np
from nexusformat import nexus as nx

class newConcat(object):
    outputFile = None
    filenames = None
    core = None
    stackItems = [
        "/entry/beams/direct/instrument/detector00/count_time",
        "/entry/beams/direct/instrument/detector00/data/original",
        "/entry/beams/direct/instrument/detector00/data/average_intensity",
        "/entry/beams/direct/instrument/detector00/data/uncertainties_poisson",
        "/entry/beams/direct/instrument/detector00/data/uncertainties_sem",
        "/entry/beams/direct/instrument/detector00/detector_readout_time",
        "/entry/beams/direct/instrument/detector00/detectorSpecific/flatfield",
        "/entry/beams/direct/instrument/detector00/detectorSpecific/pixel_mask",
        "/entry/beams/direct/instrument/detector00/det_x_encoder",
        "/entry/beams/direct/instrument/detector00/frames/data_000001",
        "/entry/beams/direct/instrument/detector00/frame_time",

        "/entry/beams/through_sample/instrument/detector00/count_time",
        "/entry/beams/through_sample/instrument/detector00/data/original",
        "/entry/beams/through_sample/instrument/detector00/data/average_intensity",
        "/entry/beams/through_sample/instrument/detector00/data/uncertainties_poisson",
        "/entry/beams/through_sample/instrument/detector00/data/uncertainties_sem",
        "/entry/beams/through_sample/instrument/detector00/detector_readout_time",
        "/entry/beams/through_sample/instrument/detector00/detectorSpecific/flatfield",
        "/entry/beams/through_sample/instrument/detector00/detectorSpecific/pixel_mask",
        "/entry/beams/through_sample/instrument/detector00/det_x_encoder",
        "/entry/beams/through_sample/instrument/detector00/frames/data_000001",
        "/entry/beams/through_sample/instrument/detector00/frame_time",

        # apparently previously linked items are not automatically re-linked???
        "/entry/data/average_intensity",
        "/entry/data/original",
        "/entry/data/uncertainties_poisson",
        "/entry/data/uncertainties_sem",
        "/entry/duration",

        "/entry/instrument/beamstop/beamstop_in",
        "/entry/instrument/beamstop/transformations/arm_length",
        "/entry/instrument/beamstop/transformations/bsr",
        "/entry/instrument/beamstop/transformations/bsz",

        "/entry/instrument/chamber_pressure",

        "/entry/instrument/collimator1/blade_bottom",
        "/entry/instrument/collimator1/blade_top",
        "/entry/instrument/collimator1/blade_negy",
        "/entry/instrument/collimator1/blade_posy",
        "/entry/instrument/collimator1/h_aperture",
        "/entry/instrument/collimator1/transformations/cen_y",
        "/entry/instrument/collimator1/transformations/cen_z",
        "/entry/instrument/collimator1/v_aperture",
        "/entry/instrument/collimator2/blade_bottom",
        "/entry/instrument/collimator2/blade_top",
        "/entry/instrument/collimator2/blade_negy",
        "/entry/instrument/collimator2/blade_posy",
        "/entry/instrument/collimator2/h_aperture",
        "/entry/instrument/collimator2/transformations/cen_y",
        "/entry/instrument/collimator2/transformations/cen_z",
        "/entry/instrument/collimator2/v_aperture",
        "/entry/instrument/collimator3/blade_bottom",
        "/entry/instrument/collimator3/blade_top",
        "/entry/instrument/collimator3/blade_negy",
        "/entry/instrument/collimator3/blade_posy",
        "/entry/instrument/collimator3/h_aperture",
        "/entry/instrument/collimator3/transformations/cen_y",
        "/entry/instrument/collimator3/transformations/cen_z",
        "/entry/instrument/collimator3/v_aperture",

        "/entry/instrument/configuration",

        '/entry/instrument/detector00/frames/data_000001',
        "/entry/instrument/detector00/beam_center_x",
        "/entry/instrument/detector00/beam_center_y",
        "/entry/instrument/detector00/count_time",
        "/entry/instrument/detector00/data/original",
        "/entry/instrument/detector00/data/average",
        "/entry/instrument/detector00/data/uncertainties_poisson",
        "/entry/instrument/detector00/data/uncertainties_sem",
        "/entry/instrument/detector00/detector_readout_time",
        "/entry/instrument/detector00/detector_module/fast_pixel_direction",
        "/entry/instrument/detector00/detector_module/slow_pixel_direction",
        "/entry/instrument/detector00/detector_module/module_offset",
        "entry/instrument/detector00/detectorSpecific/flatfield",
        "entry/instrument/detector00/detectorSpecific/pixel_mask",
        "/entry/instrument/detector00/det_x_encoder",
        "/entry/instrument/detector00/frame_time",
        "/entry/instrument/detector00/sensor_thickness",
        "/entry/instrument/detector00/threshold_energy",
        "/entry/instrument/detector00/x_pixel_size",
        "/entry/instrument/detector00/y_pixel_size",
        "/entry/instrument/detector00/transformations/euler_c",
        "/entry/instrument/detector00/transformations/euler_b",
        "/entry/instrument/detector00/transformations/euler_a",
        "/entry/instrument/detector00/transformations/det_y",
        "/entry/instrument/detector00/transformations/det_z",
        "/entry/instrument/detector00/transformations/det_x",


        # 'entry1/instrument/detector00/detector_module/data_origin', # these should not be stacked!
        # 'entry1/instrument/detector00/detector_module/data_size', # these should not be stacked!
        # in case they exist:
        "/entry/instrument/downstream_crystal/d_spacing",
        "/entry/instrument/downstream_crystal/miller_indices",
        "/entry/instrument/downstream_crystal/order_no",
        "/entry/instrument/downstream_crystal/transformations/fineyaw",
        "/entry/instrument/downstream_crystal/transformations/yaw",
        "/entry/instrument/downstream_crystal/transformations/y_translation",

        "/entry/instrument/source/current",
        "/entry/instrument/source/voltage",
        "/entry/instrument/source/transformations/dual_y",

        "/entry/instrument/upstream_crystal/d_spacing",
        "/entry/instrument/upstream_crystal/miller_indices",
        "/entry/instrument/upstream_crystal/order_no",
        "/entry/instrument/upstream_crystal/transformations/yaw",
        "/entry/instrument/upstream_crystal/transformations/y_translation",

        "/entry/sample/beam/flux",
        "/entry/sample/beam/incident_wavelength",
        "entry/sample/chamber_pressure", # already done above via a link - no it isn't... 
        "/entry/sample/thickness",
        "/entry/sample/transformations/sample_z",
        "/entry/sample/transformations/sample_y",
        "/entry/sample/transformations/sample_x",
        "/entry/sample/transmission",
        "/entry/instrument/monochromator/crystal/wavelength",
        "/entry/instrument/monochromator/wavelength_error",
        "/entry/instrument/monochromator/wavelength",
        "/entry/sample/mu", # experimental, in in-situ experiments, mu could change from sample to sample. not sure how the calculator will deal with this.
        "/entry/sample/temperature", # does not exist yet, but it will..
        # '/entry1/instrument/detector00/detectorSpecific/data_collection_date',
    ]


    def __init__(self, filenames:list = [], outputFile = None):
        assert isinstance(outputFile, Path), 'output filename must be a path instance'
        self.outputFile = outputFile
        # make sure we have an output file to write to
        self.initCore()

        # assert that the filenames to stack all exist:
        for fname in filenames:
            assert fname.exists(), f'filename {fname} does not exist in the list of files to stack.'

        # use the first file as a template
        self.createStructureFromFile(filenames[0], addShape = (len(filenames), 1))
        for idx, filename in enumerate(filenames): 
            self.addDataToStack(filename, addAtStackLocation = idx)

        self.core.close()


    def createStructureFromFile(self, ifname, addShape):
        """addShape is a tuple with the dimensions to add to the normal datasets. i.e. (280, 1) will add those dimensions to the array shape"""
        input = nx.nxload(ifname)
        # go and put everything in place, but do something special for the datasets to stack:
        for item in input.walk():
            if isinstance(item, nx.NXgroup) and item.nxpath != '/': 
                print(f'adding group: {item.nxpath}')
                self.core[item.nxpath] = nx.NXgroup(attrs = item.attrs)

            # we are only stacking these items!
            elif isinstance(item, nx.NXfield) and item.nxpath in self.stackItems: 

                print(f'* will prepare this item for stacking: {item.nxpath} with shape {np.shape(item.nxvalue)}')
                self.core[item.nxpath] = nx.NXfield(
                    shape = (*addShape, *np.shape(item.nxvalue)),
                    maxshape = (*addShape, *np.shape(item.nxvalue)),
                    dtype = item.dtype,
                    attrs=item.attrs
                )

            # we are not stacking these static items:
            elif isinstance(item, nx.NXfield) and not (item.nxpath in self.stackItems): 
                print(f'will add this static item: {item.nxpath}')
                self.core[item.nxpath] = item.copy()
            
            # did we miss any rando stuff?
            else: print(f'** uncaught object: {item.nxpath}')

    def addDataToStack(self, ifname, addAtStackLocation):
        input = nx.nxload(ifname)
        # go and put everything in place, but do something special for the datasets to stack:
        for item in input.walk():
            # we are only stacking these items!
            if isinstance(item, nx.NXfield) and item.nxpath in self.stackItems: 
                print(f'* adding data to stack: {item.nxpath} at stackLocation: {addAtStackLocation}')
                self.core[item.nxpath][addAtStackLocation] = item.nxvalue

    def initCore(self) -> None:
        # if self.outputFile.exists():
        #     self.core = nx.nxload(self.outputFile, 'r+')
        # else:
        self.core = nx.nxload(self.outputFile, 'w') # nx.NXroot(attrs={'default': 'entry', 'note': 'trial version of a USAXS data writer for 1D data from the MOUSE 20210610'})
            # self.core.attrs.update({'default': 'entry'})
            # self.core.entry=nx.NXentry(attrs={'default': 'angles'})



class NeXConcat(object):
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
    forceDType = {"entry/frames/data/data_000001": np.float64}
    stackItems = [
        "entry/instrument/beamstop/beamstop_in",
        "entry/instrument/beamstop/transformations/arm_length",
        "entry/instrument/beamstop/transformations/bsr",
        "entry/instrument/beamstop/transformations/bsz",

        "entry/instrument/chamber_pressure",

        "entry/instrument/collimator1/blade_bottom",
        "entry/instrument/collimator1/blade_top",
        "entry/instrument/collimator1/blade_negy",
        "entry/instrument/collimator1/blade_posy",
        "entry/instrument/collimator1/h_aperture",
        "entry/instrument/collimator1/transformations/cen_y",
        "entry/instrument/collimator1/transformations/cen_z",
        "entry/instrument/collimator1/v_aperture",
        "entry/instrument/collimator2/blade_bottom",
        "entry/instrument/collimator2/blade_top",
        "entry/instrument/collimator2/blade_negy",
        "entry/instrument/collimator2/blade_posy",
        "entry/instrument/collimator2/h_aperture",
        "entry/instrument/collimator2/transformations/cen_y",
        "entry/instrument/collimator2/transformations/cen_z",
        "entry/instrument/collimator2/v_aperture",
        "entry/instrument/collimator3/blade_bottom",
        "entry/instrument/collimator3/blade_top",
        "entry/instrument/collimator3/blade_negy",
        "entry/instrument/collimator3/blade_posy",
        "entry/instrument/collimator3/h_aperture",
        "entry/instrument/collimator3/transformations/cen_y",
        "entry/instrument/collimator3/transformations/cen_z",
        "entry/instrument/collimator3/v_aperture",

        "entry/instrument/configuration",

        'entry/instrument/detector00/frames/data_000001',
        "entry/instrument/detector00/beam_center_x",
        "entry/instrument/detector00/beam_center_y",
        "entry/instrument/detector00/count_time",
        "entry/instrument/detector00/data/average_intensity",
        "entry/instrument/detector00/data/uncertainties_poisson",
        "entry/instrument/detector00/data/uncertainties_sem",
        "entry/instrument/detector00/detector_readout_time",
        "entry/instrument/detector00/det_x_encoder",
        "entry/instrument/detector00/detector_module/fast_pixel_direction",
        "entry/instrument/detector00/detector_module/slow_pixel_direction",
        "entry/instrument/detector00/detector_module/module_offset",
        "entry/instrument/detector00/direct_beam_image",
        "entry/instrument/detector00/frame_time",
        "entry/instrument/detector00/sample_beam_image",
        "entry/instrument/detector00/sensor_thickness",
        "entry/instrument/detector00/threshold_energy",
        "entry/instrument/detector00/x_pixel_size",
        "entry/instrument/detector00/y_pixel_size",
        "entry/instrument/detector00/transformations/euler_c",
        "entry/instrument/detector00/transformations/euler_b",
        "entry/instrument/detector00/transformations/euler_a",
        "entry/instrument/detector00/transformations/det_y",
        "entry/instrument/detector00/transformations/det_z",
        "entry/instrument/detector00/transformations/det_x",
        # 'entry1/instrument/detector00/detector_module/data_origin', # these should not be stacked!
        # 'entry1/instrument/detector00/detector_module/data_size', # these should not be stacked!
        # in case they exist:
        "entry/instrument/downstream_crystal/d_spacing",
        "entry/instrument/downstream_crystal/miller_indices",
        "entry/instrument/downstream_crystal/order_no",
        "entry/instrument/downstream_crystal/transformations/fineyaw",
        "entry/instrument/downstream_crystal/transformations/yaw",
        "entry/instrument/downstream_crystal/transformations/y_translation",

        "entry/instrument/source/current",
        "entry/instrument/source/voltage",
        "entry/instrument/source/transformations/dual_y",

        "entry/instrument/upstream_crystal/d_spacing",
        "entry/instrument/upstream_crystal/miller_indices",
        "entry/instrument/upstream_crystal/order_no",
        "entry/instrument/upstream_crystal/transformations/yaw",
        "entry/instrument/upstream_crystal/transformations/y_translation",

        "entry/sample/beam/flux",
        "entry/sample/beam/incident_wavelength",
        # "entry/sample/chamber_pressure", # already done above via a link
        "entry/sample/thickness",
        "entry/sample/transformations/sample_z",
        "entry/sample/transformations/sample_y",
        "entry/sample/transformations/sample_x",
        "entry/sample/transmission",
        "entry/instrument/monochromator/crystal/wavelength",
        "entry/instrument/monochromator/wavelength_error",
        "entry/instrument/monochromator/wavelength",
        "entry/sample/mu", # experimental, in in-situ experiments, mu could change from sample to sample. not sure how the calculator will deal with this.
        "entry/sample/temperature", # does not exist yet, but it will..
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

        assert isinstance(self.ofname, Path), 'output filename ofname must be a Path instance'
        # delete output file if exists, and requested by input flag
        if self.ofname.exists() and self.remove:
            print(f"file {self.ofname} exists, removing...")
            self.ofname.unlink()

        for infile in self.inflist:
            assert isinstance(infile, Path), 'input filenames in input file list must be a Path instance'
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
            h5f["/"].attrs["default"] = "entry"

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
