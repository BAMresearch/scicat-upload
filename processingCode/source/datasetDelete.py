
import requests  # for HTTP requests
import json  # for easy parsing
from pathlib import Path
import h5py
import datetime
import pandas as pd
import xlrd
import urllib
# # import base64
from SAXSClasses import readLog
from scicatbam import scicatBam, h5py_casting
import logging
import time
import numpy as np
import argparse




class scicatScatterDelete(object):
    # SAXS-specific scicat scripts for uploading the datasets, proposals and samples
    # settables set in adict
    settables = ['logbookFile', 'filename', 'uploadType', 'username', 'password', 'test', 'deleteExisting']
    scb = None
    filename = None
    test = None
    logbookFile = None
    logInstance = None
    uploadType = None
    username = None
    password = None
    deleteExisting = None

    def __init__(self, scb= None, adict = None):
        # argument dictionary adict will be set as element in this class: 
        # scb is a scicatBam instance which will be used for the communication
        assert isinstance(scb, scicatBam), 'input argument scb must be an instantiated object of class scicatBam'
        assert adict is not None, 'input argument dictionary must be provided'
        self.scb = scb 

        for key, value in adict.items():
            # assert in principle superfluous as this should be checked in argparse, but we could be calling this not from the command line. 
            assert key in self.settables, f"key {key} is not a valid input argument"
            setattr(self, key, value)

        if isinstance(self.filename, str):
            self.filename = Path(self.filename.strip()) # ensure it's class Path.
        if isinstance(self.logbookFile, str):
            self.logbookFile = Path(self.logbookFile.strip()) # ensure it's class Path.
        # fill logInstance
        self.logInstance = readLog(self.logbookFile)
        self.logInstance.readMain()

    def getLbEntryFromFileName(self, filename = None):
        if filename == None: filename = self.filename
        
        YMD = filename.stem.split("_")[0]
        filenum = filename.stem.split("_")[1]
        self.year = YMD[0:4]
        self.datasetName = "" + YMD + "-" + filenum

        # logInstance = readLog(self.logbookFile)
        # lb = self.logInstance.readMain()
        self.lbEntry = (self.logInstance.getMainEntry(YMD=YMD, filenum=int(filenum))).iloc[0]
        logging.debug(f"   YMD: {YMD}, filenum: {filenum}, filename: {filename}")
        logging.debug(f"\n lbEntry: \n \t {self.lbEntry}")
        return self.year, self.datasetName, self.lbEntry


    def deleteRaw(self, filename: Path = None):
        scb = self.scb # for convenience
        if filename is None:
            filename = self.filename
        year, datasetName, lbEntry = self.getLbEntryFromFileName(filename)
        # this sets scs.year, scs.datasetName, scs.lbEntry

        logging.info(f" working on {filename}")
        # see if entry exists:
        pid = scb.getPid( # changed from "RawDatasets" to "datasets" which should be agnostic
            scb.baseurl + "datasets", {"datasetName": datasetName}, returnIfNone=0
        )
        if (pid != 0):
            # delete offending item
            url = scb.baseurl + "RawDatasets/{id}".format(id=urllib.parse.quote_plus(pid))
            scb.sendToSciCat(url, {}, cmd="delete")
            pid = 0
        

    def deleteDerived(self, filename):
        scb = self.scb # for convenience
        if filename is None:
            filename = self.filename
        year, datasetName, lbEntry = self.getLbEntryFromFileName(filename)
        # see if there is an existing derived dataset
        pid = scb.getPid(
            scb.baseurl + "datasets", {"datasetName": datasetName}, returnIfNone=0
        ) 
        if (pid != 0):
            
            # delete offending item
            url1 = scb.baseurl + "datasets/{id}".format(id=urllib.parse.quote_plus(pid))
            url2 = scb.baseurl + "DerivedDatasets/{id}".format(id=urllib.parse.quote_plus(pid))

            r1 = requests.get(url1)
            r2 = requests.get(url2)
            if r1.ok:
                scb.sendToSciCat(url1, {}, cmd="delete")
            if r2.ok:
                scb.sendToSciCat(url2, {}, cmd="delete")




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
            A script for deleting information to the SciCat database. 
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
        return parser.parse_args()

    adict = argparser()
    adict = vars(adict)

    # instantiate the uploader:
    scb = scicatBam(username=adict["username"], password=adict["password"])
    # instantiate the SAXS data uploader:
    scs = scicatScatterDelete(scb = scb, adict = adict)

    filename = Path(adict["filename"].strip())
    
    # when we have raw files:
    if adict["uploadType"] == "raw":  # raw datafile:
        #TODO:
        scs.deleteRaw(filename = filename)

    elif adict["uploadType"] == "derived":  # if set as derived
        scs.deleteDerived(filename = filename)

    elif adict["uploadType"] == "proposal":  # if "processed" in filename
        scs.deleteProposal(filename = filename)

    elif adict["uploadType"] == "samples":
        scs.deleteSample(filename = filename)

    elif adict["uploadType"] == "instrument":  # 
        scs.doeleteInstrument(filename = filename)

