# coding: utf-8
# this program generates thumbnail figures for raw and/or processed data

__author__ = "Brian R. Pauw"
__contact__ = "brian@stack.nl"
__license__ = "GPLv3+"
__date__ = "2019/11/29"
__status__ = "v1"


# Python 3 initialization:

# import h5py
import pandas
import h5py

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter 
from pathlib import Path
import argparse  # process input arguments

# from IPython.display import display, Math, Latex
matplotlib.style.use('bmh')
   
def argparser():
    parser = argparse.ArgumentParser(
        description="""
            A demonstration program showing a live camera image feed 
            alongside the intensity component of the Fourier transform
            of the camera image.\n
            Required: A python 2.7 installation (tested on Enthought Canopy),
            with: 
                - OpenCV (for camera reading)
                - numpy, TkAgg, matplotlib, scipy, argparse
            Cobbled together by Brian R. Pauw.
            Released under a GPLv3+ license.
            """
    )
    parser.add_argument(
        "-t",
        "--dataType",
        type=str,
        default='auto',
        help="can be 'raw', 'derived' or 'auto'",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="filename to process",
    )
    return parser.parse_args()


def genSciCatThumbnail(filename, dataType = 'raw'):
    
    jpgSettings= {
        'transparent' : True, 
        'dpi' : 72, 
        'optimize' : True,
        'quality' : 70,
    }

    def readHDF5Value(filename, h5Path = '/entry/result/data'):
        with h5py.File(filename, 'r') as h5f:
            return h5f[h5Path][()]

    def readHDF5Values(filename, pDict):
        outDict = {}
        dummy = [outDict.update({outKey: readHDF5Value(filename, h5Path)}) for outKey, h5Path in pDict.items()] 
        return outDict

    def pepperPlot(title = "plot title"):
        # plt.legend(loc=0)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("q (1/nm)")
        plt.ylabel(r"I (1/(m sr))")
        plt.title(title)

    # datatype can be "raw" or "derived" (i.e. corrected)
    filename = Path(filename) # just to be sure

    if dataType == "auto":
        if 'processed' in filename.stem:
            dataType = "derived"
        else:
            dataType = "raw"

    if dataType == "derived":
        pDict = {'y': '/processed/result/data',
                 'yerr': '/processed/result/errors',
                 'x' : '/processed/result/q',
                 'xerr': '/processed/result/q_errors'}
    else:
        pDict = {'y': '/entry1/instrument/detector00/data'}

    try:
        outDict = readHDF5Values(filename, pDict)
    except:
        print('could not read datasets for {} file: {}'.format(dataType, filename))
        return None

    # move to plotting stage:
    try:
        fh, (ah) = plt.subplots(1, 1, figsize = [5, 3.5])
        plt.sca(ah)

        if dataType == "derived":
            # 1D datasets for processed data
            ah = plt.errorbar(
                x = outDict['x'].flatten(), 
                y = outDict['y'].flatten(), 
                yerr = outDict['yerr'].flatten(), 
                xerr = outDict['xerr'].flatten())
            pepperPlot("")

        else:
            imDat = outDict['y']
            while imDat.ndim > 2:
                imDat = imDat.mean(axis = 0)
            imDat.clip(min = imDat[imDat > 0].min())    
            ah = plt.imshow(imDat, norm=matplotlib.colors.LogNorm())
            formatter = LogFormatter(10, labelOnlyBase=False) 
            plt.colorbar(format = formatter)

        plt.tight_layout()
        ofname = Path(filename.parent, filename.stem).as_posix() + "_thumbnail.jpg"
        plt.savefig(ofname, **jpgSettings)
        # also save as PDF, but not for storing in the database.
        plt.savefig(Path(filename.parent, filename.stem).as_posix() + "_thumbnail.pdf")
    except:
        print('could not generate thumbnail for {} file: {}'.format(dataType, filename))
        return None

    return ofname # in success, ofname is output

if __name__ == "__main__":
    # manager=pyplot.get_current_fig_manager()
    # print manager
    # process input arguments
    adict = argparser()
    # run the program, scotty! I want a kwargs object, so convert args:
    adict = vars(adict)
    genSciCatThumbnail(adict['filename'], adict['dataType'])  # and expand to kwargs