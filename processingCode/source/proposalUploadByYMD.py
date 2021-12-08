# coding: utf-8

# author: Brian R. Pauw, I. Bressler
# date: 2019.12.05
# v2.5 adds image thumbnail uploads

# Uploads the raw and processed datafiles based on the logbook and the information in the actual files.
# based on the datasetUpload_v1p1 python notebook
# ==

# we need some libraries to run these things.

import sys
import numpy as np
# import requests  # for HTTP requests
# import json  # for easy parsing
from pathlib import Path
# import h5py
# import datetime
# import pandas
import xlrd
# import hashlib
# import urllib
# import base64
from SAXSClasses import readLog
import argparse
# import xraydb
from datasetUpload import scicatBam, scicatScatter
import logging # let's try this out...
import time # trying to get rid of some overload. 

# Run this magic line to create the output script..
if __name__ == "__main__":
    logging.basicConfig(
        filename = Path(
        'debuglogs/proposalUploadByYMD_v1_runlog.log'), 
        # encoding = 'utf-8', # <-- from Python 3.9
        # filemode = 'w', # <-- if we want a new log every time.
        level = logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S'
        )
    print('step 1')
    logging.info('* * starting new log entry * *')
    def argparser():
        parser = argparse.ArgumentParser(
            description="""
            A script for uploading sample and proposal information to the SciCat database. 
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
            "-P",
            "--proposalPath",
            type=str,
            required=True,
            help="Path to where the proposal excel sheets exist",
        )
        parser.add_argument(
            "-Y",
            "--YMD",
            type=str,
            required=True,
            help="Year-month-date key to a specific logbook entry",
        )
        parser.add_argument(
            "-u",
            "--username",
            type=str,
            required=False,
            default="proposalIngestor",
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

        return parser.parse_args()

    adict = argparser()
    adict = vars(adict)
    print('step 2')
    # instantiate the uploader:
    scb = scicatBam(username=adict["username"], password=adict["password"])
    print('step 3')
    # This reimports all the proposals that are associated with one particular measurement date/ source folder
    # in this case, the "proposalPath" commandline option must be the path where the proposal excel files are stored.
    proposalPath = Path(adict["proposalPath"].strip())
    # also extract the unique proposal numbers just as an alternative to uploading a proposal file directly by filename
    logInstance = readLog(Path(adict["logbookFile"].strip()))
    print('step 4')
    # lb = logInstance.readMain()
    YMD = adict["YMD"]
    flbEntry = logInstance.getMainEntry(YMD=YMD)
    uniqueProposals = list(flbEntry.proposal.unique())
    logging.info(f'Uploading {len(uniqueProposals)} proposals for measurement date (YMD): {YMD}')
    print(f'Uploading {len(uniqueProposals)} proposals for measurement date (YMD): {YMD}')
    print(uniqueProposals)
    bdict = {
        'username': adict['username'],
        'password': adict['password'],
        'logbookFile': adict['logbookFile'],
    }
    # instantiate the SAXS data uploader:
    scs = scicatScatter(scb = scb, adict = bdict)
    print('step 5')
    logging.info('scs instantiated')
    for item in list(flbEntry.proposal.unique()):
        # find the filename
        if str(item) == '0':
            logging.warning(f'proposal not specified in logbook for measurement date (YMD): {YMD}')
            continue # empty entry
        try:
            filePath = sorted(proposalPath.glob(f'{item}*.xls*'))[0]
            logging.debug(f'using filepath: {filePath}')
        except IndexError:
            logging.warning(f'proposal with ID {item} does not exist for measurement date (YMD): {YMD}')
            logging.warning(f'globbed for files in {proposalPath} with structure {item}*.xls*')
            continue
        except:
            raise
        # now we can upload the proposal and the samples
        logging.info(f'uploading proposal with ID {item} for measurementDate (YMD): {YMD}')
        print(f'uploading proposal with ID {item} for measurementDate (YMD): {YMD}, filepath: {filePath}')
        n=1
        print(f'step 6+{n}')
        scs.doProposal(filename = filePath)
        print(f'step 7+{n}')
        scs.doSample(filename = filePath)
        print(f'step 8+{n}')
        n+=1
