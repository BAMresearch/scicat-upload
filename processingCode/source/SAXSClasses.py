# coding: utf-8

# author: Brian R. Pauw
# date: 2019.08.22

# Small classes that are used in a few of these SAXS-related scripts are stored here.

import pandas
from pathlib import Path
import h5py
import numpy as np
import os

class readLog(object):
    _logbook = None
    _logbookFile = None
    _logbookPath = None
    _collimation = None
    _dropThresh = 12  # minimum number of rows that have to be filled in
    _logbookDtypes = {
        "beamstop": "bool",
        "converttoscript": "bool",
        "filenum": "int",
        "ofgroup": "int",
        "configuration": "int",
        "nrep": "int",
        "maskdate": "datetime64[ns]",
        "date": "datetime64[ns]",
        "insitu": "bool",
        "positionx": "float",
        "positiony": "float",
        "positionz": "float",
        "sampleid": "str",
        "sampleowner": "str",
        "sampleholder": "str",
        "proposal": "int",
        "user": "str",
        "bgdate": "datetime64[ns]",
        "bgnumber": "int",
        "dbgdate": "datetime64[ns]",
        "dbgnumber": "int",
        "matrixfraction": "float", 
        "mu": "float",
        "procpipeline": "str",
        }

    _collimationDtypes = {
        "preset": "int",
        "hg1": "float",
        "hg2": "float",
        "hg3": "float",
        "vg1": "float",
        "vg2": "float",
        "vg3": "float",
        "hp1": "float",
        "hp2": "float",
        "hp3": "float",
        "vp1": "float",
        "vp2": "float",
        "vp3": "float",
        "detx": "float",
        "dety": "float",
        "detz": "float",
        "bsz": "float",
        "bsr": "float",
    }

    _logbookDefaultDict = {
        "beamstop": True,  # default values in case the logbook is not filled in completely.
        "filenum": -1,
        "ofgroup": -1,
        "configuration": 0,
        "nrep": 0,
        "maskdate": 0,
        "date": 0,
        # "sampleid" : None,
        "positionx": 0.0,
        "positiony": 0.0,
        "positionz": 0.0,
        "sampleowner": "nobody",
        "sampleholder": "undefined",
        "configuration": 0,
        "proposal": 0,
        "user": "nobody",
        "mu": -1,
        "matrixfraction": 1, 
        "procpipeline": "",
        "dbgdate": '01-01-1969',
        "dbgnumber": -1,
    }

    def __init__(self, logbookFile):
        assert (
            logbookFile is not None
        ), "I'm going to need a logbook filename before I can continue, I'm afraid... \n"
        self._logbookFile = logbookFile
        self._logbookPath = Path(self._logbookFile)
        assert (
            self._logbookPath.is_file()
        ), "The provided logbook name: {}, does not exist.".format(self._logbookFile)
        _ = self.readMain()
        _ = self.readCollimation()

    def read(self, sheetName=0, defaults=None):
        # Read in the logbook into a Pandas DataFrame. In the DataFrame format we can explore and exploit the information
        logbookSheet = pandas.read_excel(
            self._logbookPath, skiprows=2, sheet_name=sheetName
        )  # first two rows are for users
        logbookSheet = logbookSheet.dropna(
            axis=0, thresh=self._dropThresh
        )  # remove empty rows for cleaning up. Require at least 12 columns to be filled in.
        logbookSheet.columns = (
            logbookSheet.columns.str.lower()
        )  # lower case column names only
        # fill nans with defaults so we don't get trouble casting it into particular datatypes
        if defaults is not None:
            for name, default in defaults.items():
                logbookSheet[name].fillna(default, inplace=True)
        return logbookSheet

    def check(self, checkObj, dtypeDict):
        # cast into the right type
        for key, val in dtypeDict.items():
            print(
                "Casting the logbook entries from column: {} into type: {}".format(
                    key, val
                )
            )
            checkObj = checkObj.astype({key: val})
        return checkObj

    def readMain(self, sheetName=0):
        self._logbook = self.read(
            sheetName=sheetName, defaults=self._logbookDefaultDict
        )
        self._logbook = self.check(self._logbook, self._logbookDtypes)
        self._logbook[
            "YMD"
        ] = self._logbook.apply(  # add a YMD column for fast searches by readEntry
            lambda x: "{year}{month:02d}{day:02d}".format(
                year=x.date.year, month=x.date.month, day=x.date.day
            ),
            axis=1,
        )
        return self._logbook

    def readCollimation(self, sheetName="Collimation presets"):
        self._collimation = self.read(sheetName=sheetName)
        self._collimation = self.check(self._collimation, self._collimationDtypes)
        self._collimation.set_index("preset", inplace=True)
        return self._collimation

    def getMainEntry(self, YMD="", filenum=None):
        """ Reads a particular entry or series of entries, 
        based on the provided year-month-day value (i.e. 201905 for everything in may, 2019). 
        Optionally, a file number can be provided as well to return a single entry. 
        returns a filtered dataframe
        """
        if filenum is None:
            return self._logbook.loc[self._logbook.YMD.str.startswith(YMD)]
        else:
            return self._logbook.loc[
                self._logbook.YMD.str.startswith(YMD)
                & (self._logbook.filenum == int(filenum))
            ]


class NXUpdate(object):
    # class to add specific (updated) fields or datasets to a NeXus structure
    fname = None
    NeXusPath = None
    updateValue = None

    def __init__(self, fname=None, NeXusPath=None, updateValue=None):
        self.fname = fname
        self.NeXusPath = NeXusPath
        self.updateValue = updateValue
        assert self.updateValue is not None
        assert self.NeXusPath is not None
        assert self.fname is not None
        print(" # updating file {}".format(self.fname))
        print(
            "   with new value {} in path {}".format(self.updateValue, self.NeXusPath)
        )
        with h5py.File(fname, "r+") as h5f:
            h5v = h5f[NeXusPath][()]
            h5p = h5f[NeXusPath]
            if np.shape(h5v) == np.shape(updateValue):
                # straight update
                print("   Straight update")
                h5p[...] = updateValue
            elif np.ndim(updateValue) == 0:
                # singular value, cast into shape of destination
                updateValue = np.full(
                    h5v.shape, fill_value=updateValue, dtype=h5v.dtype
                )
                h5p[...] = updateValue
                print("   reshaped update")

            else:
                raise ValueError

                
class H5Base(object):
    """
    Base functionality for writing, copying and merging data in HDF5 files. To use, update the following and run:
      * initDataDict
      * groupDict
      * addAttributeDict
      * pruneList
    """

    _infile = None  # input filename
    _outfile = None  # output filename
    _ifExist = (
        "checkVersion"
    )  # Can be "check" (default, replaces if not of the latest version), "replace", or "concatenate"
    dataDict = {}
    groupDict = {}
    addAttributeDict = {}
    pruneList = []
    NeXpandVersion = None

    def __init__(
        self,
        infile=None,
        outfile=None,
        append=False,
        ifExist="check",
        NeXpandVersion="3.5",
    ):
        self._infile = infile
        self._outfile = outfile
        self._ifExist = ifExist
        self.NeXpandScriptVersion = str(NeXpandVersion)
        self.initDataDict()
        self.sanitize()

        if (self._infile is None) or (self._outfile is None):
            print("Check the usage, input and output filenames are required.")
            return
        if not self.checkOutfile():
            print("Skipping file...")
            return

        # start by copying groups:
        for src, dest in self.groupDict.items():
            self.groupCopy(src, dest)
        # then copy values and associated attributes
        # and add additional (default) values:
        for index, item in self.dataDict.iterrows():
            # uncomment this for lots of output details:
            # print("copying value in {}".format(item))
            self.valueCopy(item)
        # add remaining attributes:
        for pathKey, value in self.addAttributeDict.items():
            # print("adding attribute: {}".format(item))
            self.addAttribute(pathKey, value)
        # remove items from prunelist:
        for item in self.pruneList:
            with h5py.File(self._outfile, 'a') as h5f:
                # print("deleting item {} from outfile: {}".format(item, self.outfile))
                del h5f[item]

    def checkOutfile(self):
        """Checks whether the output file exists, and removes it if necessary.
        returns True if the processing should continue, False if it shouldn't.
        """
        exists = os.path.exists(self._outfile)
        # delete output file if exists, and requested by input flag
        if exists and self._ifExist == "replace":
            print("output file: {} exists, removing file...".format(self._outfile))
            Path(self._outfile).unlink()
            return True
        # else: remove file if it hasn't been processed wiht the right version:
        elif exists and self._ifExist == "check":
            with h5py.File(self._outfile, 'a') as h5f:
                fileversion = h5f["/"].attrs.get("NeXpand_version", default=None)
                print(
                    "file version: {} , script version: {}".format(
                        fileversion, self.NeXpandScriptVersion
                    )
                )
                if fileversion != self.NeXpandScriptVersion:  # string comparison
                    print("output file is of the wrong version, removing...")
                    Path(self._outfile).unlink()
                    return True
                else:
                    print("Output file already up-to-date, skipping...")
                    return False
        elif self._ifExist == "concatenate":
            assert exists, "When concatenating, output file must exist"
            return True
        elif not exists:
            return True

    def sanitize(self):
        self.dataDict = self.dataDict.astype(
            {"srcloc": "str", "destloc": "str", "ndmin": "float"}
        )

    def groupCopy(self, ingroup, outgroup):
        print("self.infile: {}, self.outfile: {}".format(self._infile, self._outfile))
        with h5py.File(self._outfile, 'a') as h5f, h5py.File(self._infile, "r") as h5in:
            gClass = h5in.get(ingroup, default="NonExistentGroup")
            if not isinstance(gClass, h5py.Group):
                print("group {} does not exist in input file".format(ingroup))
                return False

            # ensure output group exists:
            gid = h5f.require_group(outgroup)
            h5in.copy(ingroup, gid, expand_external=True)
        return True  # no problems encountered

    def addAttribute(self, pathKey=None, value=None):
        # adapted from Structurize:
        assert pathKey is not None, "HDF5 path/key combination cannot be empty"
        assert value is not None, "HDF5 attribute value cannot be empty"
        # print("adding attribute at pathKey {}".format(pathKey))
        path, key = pathKey.rsplit("@", 1)

        with h5py.File(self._outfile, 'a') as h5f:
            aloc = h5f.get(path, default=None)
            if aloc == "NonExistentLocation":
                print("Location {} does not exist in output file".format(aloc))
                return False

            # write attribute
            h5f[path].attrs[key] = value
        return True

    def valueCopy(self, details):

        # copy a value, its attributes, and add additional attributes.
        with h5py.File(self._outfile, 'a') as h5f, h5py.File(self._infile, "r") as h5in:
            # check if source data exists:
            gValLoc = None
            gAttrs = {}
            gVal = None
            # if 'sample_x' in details.destloc:
            #     print(f'type details: {type(details)}, details: {details}')
            # if 'sample_x' in details.destloc:
            #     print(f'before: gAttrs: {gAttrs}, details.attributes: {details.attributes}')

            if details.srcloc is not None:
                gValLoc = h5in.get(details.srcloc, default=None)
            if isinstance(gValLoc, h5py.Dataset):
                gVal = gValLoc[()]
                gAttrs = dict(gValLoc.attrs)
            else:
                # print("value {} does not exist in the input file, using default value instead"
                #       .format(details.srcloc))
                gVal = details.default

            if gVal is None:
                # print("-X- value {} does not exist in the input file, and no default value was specified. skipping..."
                #       .format(details.srcloc))
                return False

            # delete if it already exists in the output file
            if h5f.get(details.destloc, default=None) is not None:
                del h5f[details.destloc]

            if details.datatype is not None:
                # print("attempting to cast value {} into datatype: {}"
                #       .format(gVal, details.datatype))
                try:
                    gVal = details.datatype(gVal)
                except ValueError:
                    print(
                        "could not cast value {} into datatype {}, defaulting to {} for {}".format(
                            gVal, details.datatype, details.default, details.destloc
                        )
                    )
                    gVal = details.default

            if callable(details.lambfunc):
                gVal = details.lambfunc(gVal)  # apply the lambda function to the value

            # prepend the value attributes to the attribute list:
            gAttrs.update(details.attributes)  # update with values in list
            details.attributes = gAttrs  # write back for later

            if np.isfinite(details.ndmin):
                try:
                    # print("trying to add {} dimensions to value: {}".format(int(details.ndmin), gVal))
                    gVal = np.array(
                        gVal, dtype=details.datatype, ndmin=int(details.ndmin)
                    )
                except:  # didn't work...
                    pass
                    # print("-X- adding dimensions did not work: not adding dimensions to value: {}".format(gVal))
            # print("writing value: {} to location {}".format(gVal, details.destloc))

            # finally we get to the writing stage:
            # ensure group exists:
            groupLoc, datasetName = details.destloc.rsplit("/", maxsplit=1)
            gl = h5f.require_group(groupLoc)
            if isinstance(gVal, np.ndarray):
                h5f.create_dataset(details.destloc, data=gVal, compression="gzip")
            else:
                h5f.create_dataset(details.destloc, data=gVal)

            # this was used before, but can't overwrite or reshape...
            # h5in.copy(details.srcloc, gl, name = datasetName)

            # set any additional attributes:
            # if 'sample_x' in details.destloc:
            #     print(f'Items for outfile: {self._outfile}, destloc {details.destloc} from infile {self._infile}, srcloc {details.srcloc}: {details.attributes.items()}')
            for key, val in details.attributes.items():
                h5f[details.destloc].attrs[key] = val

        return True  # no problems encountered
        