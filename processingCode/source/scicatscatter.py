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
import requests  # for HTTP requests
import json  # for easy parsing
from pathlib import Path
import h5py
import datetime
import pandas as pd
import xlrd
# # import hashlib
import urllib
# # import base64
from SAXSClasses import readLog
from scicatbam import scicatBam, h5py_casting
from sciMeta import create_meta, create_short_meta, create_meta_flat
from sciCatTags import add_simple_tags
# # import argparse
import xraydb
import logging
import time
import collections.abc
import numpy as np

 
class scicatScatter(object):
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

    def uploadBit(self, pid = 0, urlAdd = None, data = None, attachFile = None):
        # upload the bits to the database
        # try sending it...
        time.sleep(1)
        print('uploadBit getting PID...')
        if not self.test and pid == 0: # and not self.uploadType in ["samples"]:
            print('uploadBit creating entry')
            logging.info("* * * * creating new entry")
            url = self.scb.baseurl + f"{urlAdd}/replaceOrCreate"
            resp = self.scb.sendToSciCat(url, data)
            #print(resp,url)
            if "pid" in resp:
                npid = resp["pid"]
            elif "id" in resp:
                npid = resp["id"]
            elif "proposalId" in resp:
                npid = resp["proposalId"]
    
        elif not self.test and pid != 0: # and not adict["uploadType"] in ["samples"]:
            print(f'uploadBit updating existing entry with pid={pid}')
            logging.info("* * * * updating existing entry")
            url1 = self.scb.baseurl + f"{urlAdd}/{urllib.parse.quote_plus(pid)}"
            url2 = self.scb.baseurl + f"{'datasets'}/{urllib.parse.quote_plus(pid)}"
            r1 = requests.get(url1)
            r2 = requests.get(url2)
            #print(r1, r2,url1)
            if r2.ok:# this should be normal behavior and this one is to fix the data type nexus
                resp = self.scb.sendToSciCat(url2, data, "patch")
            elif r1.ok:
                resp = self.scb.sendToSciCat(url1, data, "patch")
            npid = pid
               
        print('uploadBit finished updating entry')
        if attachFile is not None:
            print('uploadBit adding attachment...')
            # attach an additional file as "thumbnail"
            assert isinstance(attachFile, Path), 'attachFile must be an instance of pathlib.Path'

            if attachFile.exists():
                print('uploadBit attachfile exists...')
                logging.info("attaching thumbnail {} to {} \n".format(attachFile, npid))
                urlAddThumbnail = urlAdd
                if urlAdd == "DerivedDatasets": 
                    urlAddThumbnail="datasets" # workaround for scicat inconsistency
                print(f'uploadBit sending attachment {attachFile}...')
                resp = self.scb.addThumbnail(npid, attachFile, datasetType = urlAddThumbnail, clearPrevious=True)
                logging.debug(resp.json())
        return npid

    def doProposal(self, filename):
        scb = self.scb # for convenience
        # token = getToken(username="proposalIngestor", password="")  # get a new token
        resultDict = {}
        # read excel file:
        print('scs reading excel')
        df = pd.read_excel(
            filename,
            sheet_name=0,
            skiprows=1,
            index_col=0,
            header=None,
            names=["key", "val"],
            dtype={"key": str, "val": str},
        )
        df = df.transpose()
        df = df.astype({
            "Proposal Date": "datetime64[ns]",
            "Short Abstract": str,
            "Proposal Number": str})

        print(f'scb get PID at url {scb.baseurl + "Proposals"} for proposal number {df["Proposal Number"].val}')
        pid = scb.getPid(
            scb.baseurl + "Proposals",
            {"proposalId": df["Proposal Number"].val},
            returnIfNone=0,
            returnField = 'proposalId',
        )

        data = {
            "proposalId": df["Proposal Number"].val,
            "pi_email": df["Email"].val,
            "email": "SAXS002@bam.de",
            "pi_lastname": df["Name"].val.split(" ")[-1],
            "pi_firstname": df["Name"].val.split(" ")[:-1],
            "lastname": df["Name"].val.split(" ")[-1],
            "firstname": df["Name"].val.split(" ")[:-1],
            "title": df["Title"].val,
            "abstract": df["Short Abstract"].val,
            "createdAt": datetime.datetime.isoformat(
                (df["Proposal Date"].astype("datetime64[ns]").val)
            ),
            "updatedAt": datetime.datetime.isoformat(datetime.datetime.utcnow()) + "Z",
            "createdBy": self.username,
            "updatedBy": self.username,
            # "updatedAt": datetime.datetime.isoformat(datetime.datetime.utcnow()),
            "ownerGroup": df["Organisation"].val,
        }

        urlAdd = "Proposals"  # gives 501 if I try /replaceOrUpdate
        thumbnailFile = None # Path(filename)
        # upload
        print('scs upload bit')
        npid = self.uploadBit(pid = pid, urlAdd = urlAdd, data = data) # , attachFile = thumbnailFile)
        # not uploading the proposal itself, seems to not be available here. 
        # print('scs upload proposal itself')
        # logging.info("* * * * adding datablock")
        # self.scb.addDataBlock(npid, filename, datasetType='datasets', clearPrevious=True)

    def doRaw(self, filename: Path = None):
        scb = self.scb # for convenience
        if filename is None:
            filename = self.filename
        year, datasetName, lbEntry = self.getLbEntryFromFileName(filename)
        # this sets scs.year, scs.datasetName, scs.lbEntry

        sciMetadataKeys = [
            "/entry1/frames/count_time",
            "/entry1/instrument/detector00/count_time",
            "/entry1/instrument/detector00/detector_number",
            "/entry1/instrument/detector00/threshold_energy",
            "/entry1/instrument/detector00/transformations/det_x",
            "/entry1/instrument/detector00/det_x_encoder",
            "/entry1/instrument/detector00/transformations/det_y",
            "/entry1/instrument/detector00/transformations/det_z",
            "/entry1/sample/transformations/sample_x",
            "/entry1/sample/transmission",
            "/entry1/sample/thickness",
            "/entry1/sample/beam/flux",
            "/entry1/sample/beam/incident_wavelength",
            "/entry1/sample/sampleholder",
            "/entry1/instrument/configuration",
        ]
        """ OLD VERSION
        sciMetadataKeyDict = {key: key for key in sciMetadataKeys}
        scb.h5GetDict(filename, sciMetadataKeyDict) 
        """
        #sciMeta = create_short_meta(filename, sciMetadataKeys) # Option 1: short meta for the predefined list of keys
        sciMeta = create_meta(filename)                        # Option 2: all Meta json-like. Not convenient to search 
        #sciMeta = create_meta_flat(filename)                   # Option 3: all Meta flat. Better searchable

        logging.info(f" working on {filename}")
        if str(lbEntry.sampleid).startswith("{}".format(year)):
            sampleId = str(lbEntry.sampleid)
        else:
            sampleId = scb.h5Get(filename, "/entry1/sample/name")
        # see if entry exists:
        pid = scb.getPid( # changed from "RawDatasets" to "datasets" which should be agnostic
            scb.baseurl + "datasets", {"datasetName": datasetName}, returnIfNone=0
        )
        if (pid != 0) and self.deleteExisting:
            # delete offending item
            url = scb.baseurl + "RawDatasets/{id}".format(id=urllib.parse.quote_plus(pid))
            scb.sendToSciCat(url, {}, cmd="delete")
            pid = 0
        data = {  # model for the raw datasets as defined in the RawDatasets
            "owner": scb.h5Get(filename, "/entry1/user"),
            "contactEmail": "brian.pauw@bam.de",
            "createdBy": self.username,
            "updatedBy": self.username,
            "creationLocation": "SAXS002",
            "creationTime": datetime.datetime.isoformat(datetime.datetime.utcfromtimestamp(filename.stat().st_mtime)) ,
            # "creationTime": scb.h5Get(filename, "/Saxslab/start_timestamp"), # SAXSlab doesn't exist anymore... 
            "updatedAt": datetime.datetime.isoformat(datetime.datetime.utcnow()) + "Z",
            # "createdAt": datetime.datetime.isoformat(datetime.datetime.utcnow()) + "Z",
            # "creationTime": h5Get(filename, "/processed/process/date"),
            "datasetName": datasetName,
            "type": "raw",
            "instrumentId": "SAXS002",
            "keywords": add_simple_tags(filename),
            "ownerGroup": "BAM 6.5",
            "accessGroups": ["BAM", "BAM 6.5"],
            "proposalId": str(lbEntry.proposal),
            "dataFormat": "NeXus",
            "principalInvestigator": scb.h5Get(filename, "/entry1/sample/sampleowner"),
            "pid": pid,
            "size": 0,
            "sourceFolder": filename.parent.as_posix(),
            "size": scb.getFileSizeFromPathObj(filename),
            "scientificMetadata": sciMeta,
            "sampleId": str(sampleId),
        }
        urlAdd = "RawDatasets"
        
        # determine thumbnail: 
        thumbnailFile = Path(Path(filename.parent, filename.stem).as_posix() + "_thumbnail.jpg")
        # upload
        if thumbnailFile.exists():
            npid = self.uploadBit(pid = pid, urlAdd = urlAdd, data = data, attachFile = thumbnailFile)
        logging.info("* * * * adding datablock")
        self.scb.addDataBlock(npid, filename, datasetType='datasets', clearPrevious=False)


    def doDerived(self, filename):
        scb = self.scb # for convenience
        if filename is None:
            filename = self.filename
        year, datasetName, lbEntry = self.getLbEntryFromFileName(filename)
    
        # add processed data as DerivedDatasets
        rawDatasetName = (
            "" + filename.stem.split("_")[0] + "-" + filename.stem.split("_")[1]
        )
        rawPid = scb.getPid( # get the original dataset it belongs to
            scb.baseurl + "RawDatasets",
            {"datasetName": rawDatasetName},
            returnIfNone=0,
        )
        if rawPid == 0:
            logging.error("cannot add derived dataset, raw dataset does not exist")
            raise ValueError
        
        inputDatasets = [rawPid]
        # now try adding the Pids for the background and the dispersantbackground:
        for idSet in ["/entry1/backgroundfilename", "/entry1/dispersantbackgroundfilename"]:
            xfname = Path(scb.h5Get(filename, idSet, default = '')) # set a default we can parse in the next line
            if not "_" in xfname.stem: continue # probably doesn't exist, so try the next one
            # find the exact match:
            exfname = ("" + xfname.stem.split("_")[0] + "-" + xfname.stem.split("_")[1])
            xPid = scb.getPid( # get the original dataset it belongs to
                scb.baseurl + "RawDatasets",
                {"datasetName": exfname },
                returnIfNone=0,
            )
            if xPid != 0:
                inputDatasets += [xPid]

        logging.info("* * * * * ID retrieved of raw dataset: {}".format(rawPid))
        # raw dataset available
        datasetName = rawDatasetName + "-corrected"
        # see if there is an existing derived dataset
        pid = scb.getPid(
            scb.baseurl + "datasets", {"datasetName": datasetName}, returnIfNone=0
        ) # datasets works for raw and derived..
        if (pid != 0) and self.deleteExisting:
            
            # delete offending item
            url1 = scb.baseurl + "datasets/{id}".format(id=urllib.parse.quote_plus(pid))
            url2 = scb.baseurl + "DerivedDatasets/{id}".format(id=urllib.parse.quote_plus(pid))
            r1 = requests.get(url1)
            r2 = requests.get(url2)
            if r1.ok:
                scb.sendToSciCat(url1, {}, cmd="delete")
            if r2.ok:
                scb.sendToSciCat(url2, {}, cmd="delete")

            pid = 0

        creationTime = scb.h5Get(filename, "/entry/process/date", default = None)
        logging.debug("creation time: {}".format(creationTime))
        if creationTime == None:
            creationTime = scb.h5Get(filename, "/processed/process/date")
            logging.debug("creation time: {}".format(creationTime))
        # construct scientific metadata
        sciMeta = create_meta(filename) 
        # construct data
        data = {
            "investigator": scb.h5Get(filename, "/entry1/sample/sampleowner"),
            # "inputDatasets": [ "RawDatasets/{id}".format(id=urllib.parse.quote_plus(rawPid))],
            "inputDatasets": inputDatasets,
            "usedSoftware": [
                scb.h5Get(filename, "/processed/process/program")
                + " "
                + scb.h5Get(filename, "/processed/process/version")
            ],
            "jobParameters": {},
            "jobLogData": "",
            "scientificMetadata": sciMeta,
            "owner": scb.h5Get(filename, "/entry1/user"),
            "ownerEmail": "",
            "orcidOfOwner": "",
            "contactEmail": "brian.pauw@bam.de",
            "sourceFolder": filename.parent.as_posix(),
            "size": scb.getFileSizeFromPathObj(filename),
            "packedSize": scb.getFileSizeFromPathObj(filename),
            "creationTime": creationTime,
            "type": "derived",
            "validationStatus": "valid",
            "instrumentId": "SAXS002",
            "keywords": [""],
            "description": "Dataset corrected with Dawn",  # .format(resp[0]["datasetName"]),
            "datasetName": datasetName,
            "classification": "string",
            "license": "CC-BY",
            "version": "3.0",
            "isPublished": False,
            "ownerGroup": "BAM 6.5",
            "accessGroups": ["BAM", "BAM 6.5"],
            # "proposalId": str(lbEntry.proposal),
            "createdBy": self.username,
            "updatedBy": self.username,
            # "createdAt": getFileModTimeFromPathObj(filename),
            "updatedAt": datetime.datetime.isoformat(datetime.datetime.utcnow()) + "Z"
            
            #       "updatedAt": "2019-07-22T14:55:49.440Z",
        }
        urlAdd = "DerivedDatasets"
        
        # determine thumbnail: 
        thumbnailFile = Path(Path(filename.parent, filename.stem).as_posix() + "_thumbnail.jpg")
        # upload
        npid = self.uploadBit(pid = pid, urlAdd = urlAdd, data = data, attachFile = thumbnailFile)
        logging.info("* * * * adding datablock")
        scb.addDataBlock(npid, filename, datasetType='datasets', clearPrevious=False)

    def doInstrument(self, filename):
        scb = self.scb # for convenience

        # token = getToken(username = 'ingestor', password = '') # get a new token
        resultDict = {}
        # read excel file:
        df = pd.read_excel(filename, 1, index_col=0, 
                            header=None, names = ['value'])
        df = df.rename(columns={'value': df.loc['id','value']})
        df = df.drop(['document', 'version'])
        df = df.transpose().iloc[0,:].to_dict()
        for key in df.keys():
            if key in ["start_date", "last_update", "end_date"]:
                try:
                    df[key] = datetime.datetime.isoformat(datetime.datetime.strptime(str(df[key])[:10], '%Y-%m-%d')) + "Z"
                except:
                    df[key] = str(df[key])
            elif key in ["instrument_class"]:
                df[key] = int(df[key])
            elif key in ['room_telephone', 'resp_telephone', "instrument_class", "serial_number", "inventory_number"]:
                try:
                    df[key] = int(df[key])
                except:
                    df[key] = str(df[key])
            else:
                df[key] = str(df[key])

        pid = scb.getPid(
            scb.baseurl + "Instruments",
            {"pid": df["id"]},
            returnIfNone=0,
            returnField = 'pid',
        )

        data = {
            "pid": df["id"],
            "name": df["name"],
            "customMetadata": {
                **df
            }
        }

        urlAdd = "Instruments"  # gives 501 if I try /replaceOrUpdate

        npid = self.uploadBit(pid = pid, urlAdd = urlAdd, data = data, attachFile = None)

    def doSample(self, filename):
        def gen_dict_extract(key, var): # from stackoverflow..
            if hasattr(var,'items'):
                for k, v in var.items():
                    if k == key:
                        yield v
                    if isinstance(v, dict):
                        for result in gen_dict_extract(key, v):
                            yield result
                    # elif isinstance(v, list):
                    #     for d in v:
                    #         for result in gen_dict_extract(key, d):
                    #             yield result

        def genSampleComponent(row):
            sCompKeys = [  # for setting datatypes of the logbook columns
                "componentId",
                "componentSurroundedBy",
                "componentName",
                "composition",
                "density",
                "volFrac",
                "massFrac",
                "associatedHazards",
                "MSDSAvailable",
                "eLogId",
            ]
            sComp = dict.fromkeys(sCompKeys)
            sComp.update(row.loc[sCompKeys].to_dict())
            # try adding an X-ray absorption coefficient
            try:
                logging.debug(f'composition: {sComp["composition"]}, density: {sComp["density"]}, mu: { xraydb.material_mu(sComp["composition"], 8.04e3, density = sComp["density"], kind = "total" )}')
                # kind can be adjusted to 'photo' to calculate just the photo-absorption... calculate not in 1/cm but in 1/m.. SI please.
                sComp.update({"absCoeffCuTotal": xraydb.material_mu(sComp['composition'], 8.04e3, density = sComp['density'], kind = 'total' ) * 100})
                sComp.update({"absCoeffCuPhoto": xraydb.material_mu(sComp['composition'], 8.04e3, density = sComp['density'], kind = 'photo' ) * 100}) 
                sComp.update({"absCoeffMoTotal": xraydb.material_mu(sComp['composition'], 17.4e3, density = sComp['density'], kind = 'total' ) * 100})
                sComp.update({"absCoeffMoPhoto": xraydb.material_mu(sComp['composition'], 17.4e3, density = sComp['density'], kind = 'photo' ) * 100})
            except:
                logging.warning('X-ray absorption coefficient could not be calculated')
            return sComp

        def genSample(sI):
            """ generates a dictionary with nested sample components. The main properties of this sample entry are calculated afterward. 
            This function should be provided a set of rows from the sample entry form in pandas form, all belonging to the same sampleId. 
            - The volume fraction of the sample entity is always 1, the components may take partial volume fractions. 
            - The mass density, composition, etc. is the volume-weighted mean density of the components, unless an overall values are provided
            """
            assert len(sI.sampleId.unique())==1, 'only one sample should be fed into the genSample function'
            sampleKeys = [  # for setting datatypes of the logbook columns
                "sampleId",
                "sampleName",
                "composition",
                "density",
                "volFrac",
                "massFrac",
                "associatedHazards",
                "MSDSAvailable",
                "storageRequirements",
                "preparationRequirements",
                "sampleNotes",
                "eLogId",
            ]
            
            sample = dict.fromkeys(sampleKeys)
            sumVfAbsCoeffCuPhoto = 0.0 # initialize these values to zero, fractional component contributions will be added
            sumVfAbsCoeffMoPhoto = 0.0 # same same
            totalVf = 0.0 # this should end up as 1, otherwise renormalize X-ray absorption coefficient
            totalDensity = 0.0 # will be divided later by the volume fraction
            totalComposition = '' # this will be more tricky.. let's see if this works. 

            for rowi, row in sI.iterrows():
                # if the componentId and surroundedby fields are empty, we are talking about the main sample here
                if (row.componentId == '') and (row.componentSurroundedBy == ''): # also run this in the case of a single-row description. 
                    sample.update(row.loc[sampleKeys].to_dict())
                    continue 
                if (len(sI.index)==1): # also run this in the case of a single-row description. 
                    # sample.update(row.loc[sampleKeys].to_dict())
                    rowDict = row.loc[sampleKeys].to_dict() 
                    sample.update( 
                        {k: rowDict[k] for k in ('sampleId', 'sampleName', 'eLogId', 'sampleNotes', 'preparationRequirements', 'storageRequirements', 'associatedHazards', 'MSDSAvailable') if k in rowDict}
                        )
                    # but now don't go to the next row...
                # for now, we assume that the components are organized in a logical manner, with the surrounding component first. If not, we'll have to do some reorganization which we will skip for now...
                if (row.componentId != ''):
                    subDict = None
                    if (row.componentSurroundedBy == ''):
                        # part of main component
                        subDict = sample
                    else: 
                        # find the encapsulating component:
                        subDict = next(gen_dict_extract(f'component_{row.componentSurroundedBy}', sample))
                    # make sure the encapsulating component exists:
                    assert subDict is not None, 'surrounding component not defined yet'
                    # make a sample component
                    sComp = genSampleComponent(row)
                    subDict.update({f'component_{row.componentId}': sComp})
                    # add to the total counters for this sample:
                    try:
                        totalVf += sComp['volFrac']
                        totalDensity += sComp['density'] * sComp['volFrac']
                        totalComposition += f'({sComp["composition"]}){sComp["volFrac"]}'
                        sumVfAbsCoeffCuPhoto += sComp['volFrac'] * sComp['absCoeffCuPhoto']
                        sumVfAbsCoeffMoPhoto += sComp['volFrac'] * sComp['absCoeffMoPhoto']
                    except KeyError:
                        logging.warning('X-ray data could not be derived, probably incomplete data in sample sheet.')
                    except:
                        raise

                try:
                    # normalize by total volume fraction
                    sample['sumVf'] = totalVf
                    # sample['volFrac']
                    sample['sumVfDensity'] = totalDensity / totalVf
                    sample['sumVfComposition'] = totalComposition
                    sample['sumVfAbsCoeffCuPhoto'] = sumVfAbsCoeffCuPhoto / totalVf
                    sample['sumVfAbsCoeffMoPhoto'] = sumVfAbsCoeffMoPhoto / totalVf
                except:
                    logging.warning('overall sample composition information could not be computed, probably due to incomplete data in sample sheet')
                
                
            return sample


        scb = self.scb
        dtypeDict = {  # for setting datatypes of the logbook columns
            "sampleId": "int",
            "sampleName": "str",
            "composition": "str",
            "density": "float",
            "volFrac": "float",
            "massFrac": "float",
            "componentId": "str",
            "componentSurroundedBy": "str",
            "componentName" : "str",
            "associatedHazards": "str",
            "MSDSAvailable": "bool",
            "storageRequirements": "str",
            "preparationRequirements": "str",
            "sampleNotes": "str",
            "eLogId" : "str",
        }
        defaultDict = {
            "sampleId": -1,
            "sampleName": "",
            "composition": "",
            "density": -1,
            "volFrac": -1,
            "massFrac": -1,
            "componentId": "",
            "componentSurroundedBy": "",
            "componentName" : "",
            "associatedHazards": "",
            "MSDSAvailable": False,
            "storageRequirements": "No special storage needs",
            "preparationRequirements": "No special preparation needs",
            "sampleNotes" : "",
            "eLogId" : "",
        }

        proposal = pd.read_excel(
            filename,
            sheet_name=0,
            skiprows=1,
            index_col=0,
            header=None,
            names=["key", "val"],
            dtype={"key": str, "val": str},
        )
        proposal = proposal.transpose()
        proposal = proposal.astype({"Proposal Date": "datetime64[ns]"})


        sampleInfo = pd.read_excel(
            filename, skiprows=2, sheet_name="Sample_Info"
        )  # first two rows are for users

        sampleInfo = sampleInfo.dropna(
            axis=0, thresh=1, # how="all", subset=["sampleName"]
        )  # remove empty rows for cleaning up.

        # fill nans with defaults so we don't get trouble casting it into particular datatypes
        for name, default in defaultDict.items():
            sampleInfo[name].fillna(default, inplace=True)

        # cast into the right type
        sampleInfo = sampleInfo.astype(dtypeDict, errors = 'ignore')

        # strip leading and trailign whitespaces off of the sample composition:
        sampleInfo.composition = sampleInfo.composition.apply(lambda x: x.strip() if isinstance(x, str) else x)
        # also interposed whitespaces:
        sampleInfo.composition = sampleInfo.composition.apply(lambda x: x.replace(' ', '') if isinstance(x, str) else x)

        # fill in missing sampleId values:
        previousSampleId = -1
        for rowIdx, row in sampleInfo.iterrows():
            if row.sampleId != -1:
                previousSampleId = row.sampleId # we found a row where a new sample has been filled in
            else:
                if previousSampleId == -1 : continue # skip this row, it's before the start of the first sample
                sampleInfo.loc[rowIdx, 'sampleId'] = previousSampleId

        # cycle through the sample list to extract missing information
        for sId in sampleInfo.sampleId.unique():
            if sId == -1 : continue # skip any -1 values
            # start by filtering all components of just that sample:
            sI = sampleInfo.loc[sampleInfo.sampleId == sId].copy()
            res = genSample(sI)
            # upload the examples
            infoDict = {
                "accessGroups": ["BAM 6.5", proposal["Organisation"].val],
                "sampleId": "{}-{}".format(
                    proposal["Proposal Number"].val, res["sampleId"]
                ),
                "owner": proposal["Name"].val,
                "description": res["sampleName"],
                "createdAt": datetime.datetime.isoformat(
                    (proposal["Proposal Date"].astype("datetime64[ns]").val)
                ),
                "sampleCharacteristics": res,
                "ownerGroup": proposal["Organisation"].val,
                "accessGroups": [proposal["Organisation"].val],
                "createdBy": "brian",
                "updatedBy": "brian",
                "updatedAt": datetime.datetime.isoformat(datetime.datetime.utcnow()) + "Z",
            }
            
            logging.debug(json.dumps(infoDict))
            # try sending it... Uploading multiple samples, so cannot be done by the below bit...
            url = scb.baseurl + "Samples/replaceOrCreate"  
            logging.debug(url)
            scb.sendToSciCat(url, infoDict)
