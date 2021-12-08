# coding: utf-8

# author: Brian R. Pauw, I. Bressler
# date: 2019.12.05
# v2.5 adds image thumbnail uploads

# Uploads the raw and processed datafiles based on the logbook and the information in the actual files.
# based on the datasetUpload_v1p1 python notebook
# ==

# we need some libraries to run these things.

# import sys
# import numpy as np
# import requests  # for HTTP requests
# import json  # for easy parsing
from pathlib import Path
# # import h5py
# import datetime
# import pandas
# import xlrd
# # import hashlib
# import urllib
# # import base64
# from SAXSClasses import readLog
from scicatbam import scicatBam
from scicatscatter import scicatScatter
import argparse
# import xraydb
import logging


# Run this magic line to create the output script..
if __name__ == "__main__":

    logging.basicConfig(
        filename = Path(
        'debuglogs/datasetUpload_runlog.log'), 
        # encoding = 'utf-8', # <-- from Python 3.9
        # filemode = 'w', # <-- if we want a new log every time.
        level = logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S'
        )

    def argparser():
        parser = argparse.ArgumentParser(
            description="""
            A script for uploading information to the SciCat database. 
            """
        )
        parser.add_argument(
            "-i",
            "--logbookFile",
            type=str,
            required=True,
            help="Input excel measurement logbook",
        )
        parser.add_argument(
            "-f", "--filename", type=str, required=True, help="filename to upload"
        )
        parser.add_argument(
            "-t",
            "--uploadType",
            type=str,
            required=True,
            choices=["raw", "derived", "proposal", "samples", 'instrument'],
            default="raw",
            help="Must be one of 'raw', 'derived', 'proposal', 'samples', 'instrument' ",
        )
        parser.add_argument(
            "-u",
            "--username",
            type=str,
            required=False,
            default="ingestor",
            help="database user username ",
        )
        parser.add_argument(
            "-p",
            "--password",
            type=str,
            required=False,
            default="",
            help="database user password ",
        )
        parser.add_argument(
            "-T", "--test", action="store_true", help="does not upload anything "
        )
        parser.add_argument(
            "-D",
            "--deleteExisting",
            action="store_true",
            help="if the file already exists in the database, deletes before uploading (can be used to DB errors)",
        )
        return parser.parse_args()

    adict = argparser()
    adict = vars(adict)

    # instantiate the uploader:
    scb = scicatBam(username=adict["username"], password=adict["password"])
    # instantiate the SAXS data uploader:
    scs = scicatScatter(scb = scb, adict = adict)

    filename = Path(adict["filename"].strip())
    
    # when we have raw files:
    if adict["uploadType"] == "raw":  # raw datafile:
        #TODO:
        scs.doRaw(filename = filename)

    elif adict["uploadType"] == "derived":  # if set as derived
        scs.doDerived(filename = filename)

    elif adict["uploadType"] == "proposal":  # if "processed" in filename
        scs.doProposal(filename = filename)

    elif adict["uploadType"] == "samples":
        scs.doSample(filename = filename)

    elif adict["uploadType"] == "instrument":  # 
        scs.doInstrument(filename = filename)



    # Pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    # mapParam = [item for index, item in lbEntry.iterrows()]
    # rawData = Pool.map(runRow, mapParam)
    # Pool.close()
    # Pool.join()
