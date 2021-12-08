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
import requests  # for HTTP requests
import json  # for easy parsing
from pathlib import Path
import h5py
import datetime
# import pandas
# import xlrd
import hashlib
import urllib
import base64
# from SAXSClasses import readLog
# import argparse
# import xraydb
import logging
import OpenSSL
import urllib3

def h5py_casting(val, leaveAsArray = False):
    if isinstance(val, np.ndarray) and (not leaveAsArray):
        if val.size == 1:
            val = np.array([val.squeeze()])[0]
        else:
            if np.isnan(val).sum() + np.isinf(val).sum()==np.prod([i for i in val.shape]):
                #print('all elements are either nan or inf')
                val = '-'
            elif np.isnan(val.mean()):
                #print('nan pixel at index', np.argwhere(np.isnan(val)))
                val = np.nanmean(val)
            else:
                val = val.mean()
    if isinstance(val, float) and np.isnan(val):
        val = '-'
    if isinstance(val, np.bytes_) or isinstance(val, bytes):
        val = val.decode('UTF-8')
    if isinstance(val, np.generic):
        val = val.item()    
    if isinstance(val, str):
        if val[:2]=="b'":
            val = val[2:-1]
    return val

class scicatBam(object):
    # settables
    host = "api.scicat.tld"
    baseurl = "https://" + host + "/api/v3/"
    timeouts = (20, 20)  # we are hitting a transmission timeout...
    # timeouts = None  # we are hitting a transmission timeout...
    sslVerify = True # do not check certificate
    username="ingestor"
    password=""
    retries=5 # maximum number of retries when the server freezes
    requestHeaders = {'Content-type': 'application/json', 'Accept': 'application/json'} # trying to force headers to get more reliable response
    # You should see a nice, but abbreviated table here with the logbook contents.
    token = None # store token here
    settables = ['host', 'baseurl', 'timeouts', 'sslVerify', 'username', 'password', 'token']
    pid = None # gets set if you search for something
    entries = None # gets set if you search for something
    datasetType = "RawDatasets"
    datasetTypes = ["RawDatasets", "DerivedDatasets", "Proposals"]


    def __init__(self, **kwargs):
        # nothing to do
        logging.info("initializing scicatbam...")
        for key, value in kwargs.items():
            assert key in self.settables, f"key {key} is not a valid input argument"
            setattr(self, key, value)
        # get token
        self.token = self.getToken(username=self.username, password=self.password)

    def getToken(self, username=None, password=None):
        if username is None: username = self.username
        if password is None: password = self.password
        """logs in using the provided username / password combination and receives token for further communication use"""
        logging.info("Getting new token ...")
        logging.info(f'getting token from url: {self.baseurl + "Users/login"}')
        ntry=0
        success = False
        while (ntry < self.retries) and not success:
            try:
                response = requests.post(
                    self.baseurl + "Users/login",
                    json={"username": username, "password": password},
                    timeout=self.timeouts,
                    stream=False,
                    verify=self.sslVerify,
                )
                success = True
                ntry +=1
            except (OpenSSL.SSL.WantReadError, urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout) as e:
                # sometimes, the server doesn't respond. 
                logging.error(f'server irresponsive, trial {ntry}, trying again if less than {self.retries}')
                ntry +=1
            except: raise            

        if not response.ok:
            logging.error(f'** Error received: {response}')
            err = response.json()["error"]
            logging.error(f'{err["name"]}, {err["statusCode"]}: {err["message"]}')
            sys.exit(1)  # does not make sense to continue here
            data = response.json()
            logging.error(f"Response: {data}")

        data = response.json()
        logging.info("Response:", data)
        token = data["id"]  # not sure if semantically correct
        logging.info(f"token: {token}")
        self.token = token # store new token
        return token


    def sendToSciCat(self, url, dataDict = None, cmd="post"):
        """Sends a command to the SciCat API server using url and token, returns the response JSON
        Get token with the getToken method"""
        logging.info(f'** sendToSciCat, url: {url}, access token: {self.token}, command: {cmd}, data: {json.dumps(dataDict)}')
        ntry=0
        success = False
        while (ntry < self.retries) and not success:
            try:        
                if cmd == "post":
                    response = requests.post(
                        url,
                        params={"access_token": self.token},
                        json=dataDict,
                        timeout=self.timeouts,
                        stream=False,
                        verify=self.sslVerify,
                        headers=self.requestHeaders,
                    )
                elif cmd == "delete":
                    response = requests.delete(
                        url, params={"access_token": self.token}, 
                        timeout=self.timeouts, 
                        stream=False,
                        verify=self.sslVerify,
                        headers=self.requestHeaders,
                    )
                elif cmd == "get":
                    response = requests.get(
                        url,
                        params={"access_token": self.token},
                        json=dataDict,
                        timeout=self.timeouts,
                        stream=False,
                        verify=self.sslVerify,
                        headers=self.requestHeaders,
                    )
                elif cmd == "patch":
                    response = requests.patch(
                        url,
                        params={"access_token": self.token},
                        json=dataDict,
                        timeout=self.timeouts,
                        stream=False,
                        verify=self.sslVerify,
                        headers=self.requestHeaders,
                    )
                success = True
                ntry +=1
            except (OpenSSL.SSL.WantReadError, urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout) as e:
                # sometimes, the server doesn't respond. 
                logging.error(f'server irresponsive, trial {ntry}, trying again if less than {self.retries}')
                ntry +=1
            except: raise    
        rdata = response.json()

        # print(response.url)
        if not response.ok:
            err = response.json()["error"]
            logging.error(f'{err["name"]}, {err["statusCode"]}: {err["message"]}')
            logging.error("returning...")
            rdata = response.json()
            logging.error(f"Response: {json.dumps(rdata, indent=4)}")
        # force close connection
        response.close()
        
        return rdata

    def h5Get(self, filename, h5path, default = 'none', leaveAsArray = False):
        """ get a single value from an HDF5 file, with some error checking and default handling"""
        with h5py.File(filename, "r") as h5f:
            try:
                val = h5f.get(h5path)[()]
                val = h5py_casting(val, leaveAsArray)# sofya added this line
                # logging.info('type val {} at key {}: {}'.format(val, h5path, type(val)))
                '''
                sofya commented this piece out
                if isinstance(val, np.ndarray) and (not leaveAsArray):
                    if val.size == 1:
                        val = np.array([val.squeeze()])[0]
                    else:
                        val = val.mean()
                if isinstance(val, np.int32):
                    val = int(val)  # this could go wrong...
                if isinstance(val, np.float32):
                    val = float(val)'''

            except TypeError:
                logging.warning("cannot get value from file path {}, setting to default".format(h5path))
                val = default
        return val

    def h5GetDict(self, filename, keyPaths):
        """creates a dictionary with results extracted from an HDF5 file"""
        resultDict = {}
        for key, h5path in keyPaths.items():
            resultDict[key] = self.h5Get(filename, key) # this probably needs to be key, not h5path
        return resultDict


    def getFileModTimeFromPathObj(self, pathobj):
        # may only work on WindowsPath objects...
        # timestamp = pathobj.lstat().st_mtime
        return str(datetime.datetime.fromtimestamp(pathobj.lstat().st_mtime))


    def getFileSizeFromPathObj(self, pathobj):
        filesize = pathobj.lstat().st_size
        return filesize


    def getFileChecksumFromPathObj(self, pathobj):
        with open(pathobj) as file_to_check:
            # pipe contents of the file through
            return hashlib.md5(file_to_check.read()).hexdigest()

    def clearPreviousAttachments(self, datasetId, datasetType):
        # remove previous entries to avoid tons of attachments to a particular dataset. 
        # todo: needs appropriate permissions!
        self.getEntries(url = self.baseurl + "Attachments", whereDict = {"datasetId": str(datasetId)})
        for entry in self.entries:
            url = self.baseurl + f"Attachments/{urllib.parse.quote_plus(entry['id'])}"
            self.sendToSciCat(url, {}, cmd="delete")

    def addDataBlock(self, datasetId = None, filename = None, datasetType="RawDatasets", clearPrevious = False):
        if clearPrevious:
            self.clearPreviousAttachments(datasetId, datasetType)

        dataBlock = {
            # "id": pid,
            "size": self.getFileSizeFromPathObj(filename),
            "dataFileList": [
                {
                    "path": str(filename.absolute()),
                    "size": self.getFileSizeFromPathObj(filename),
                    "time": self.getFileModTimeFromPathObj(filename),
                    "chk": "",  # do not do remote: getFileChecksumFromPathObj(filename)
                    "uid": str(
                        filename.stat().st_uid
                    ),  # not implemented on windows: filename.owner(),
                    "gid": str(filename.stat().st_gid),
                    "perm": str(filename.stat().st_mode),
                }
            ],
            "ownerGroup": "BAM 6.5",
            "accessGroups": ["BAM", "BAM 6.5"],
            "createdBy": "datasetUpload",
            "updatedBy": "datasetUpload",
            "datasetId": datasetId,
            "updatedAt": datetime.datetime.isoformat(datetime.datetime.utcnow()) + "Z",
            "createdAt": datetime.datetime.isoformat(datetime.datetime.utcnow()) + "Z",
            # "createdAt": "",
            # "updatedAt": ""
        }
        
        url = self.baseurl + f"{datasetType}/{urllib.parse.quote_plus(datasetId)}/origdatablocks"
        logging.debug(url)
        resp = self.sendToSciCat(url, dataBlock)
        return resp


    def getEntries(self, url, whereDict = {}):
        print(url, whereDict)
        # gets the complete response when searching for a particular entry based on a dictionary of keyword-value pairs
        resp = self.sendToSciCat(url, {"filter": {"where": whereDict}}, cmd="get")
        self.entries = resp
        return resp


    def getPid(self, url, whereDict = {}, returnIfNone=0, returnField = 'pid'):
        # returns only the (first matching) pid (or proposalId in case of proposals) matching a given search request
        print(f'wheredict: {whereDict}')
        resp = self.getEntries(url, whereDict)
        if resp == []:
            # no raw dataset available
            pid = returnIfNone
        else:
            #print(f'resp: {resp}')
            pid = resp[0][returnField]
        self.pid = pid
        return pid
        
    def addThumbnail(self, datasetId = None, filename = None, datasetType="RawDatasets", clearPrevious = False):
        if clearPrevious:
            self.clearPreviousAttachments(datasetId, datasetType)

        logging.info('addThumbnail: encoding file')
        def encodeImageToThumbnail(filename, imType = 'jpg'):
            header = "data:image/{imType};base64,".format(imType=imType)
            with open(filename, 'rb') as f:
                data = f.read()
            dataBytes = base64.b64encode(data)
            dataStr = dataBytes.decode('UTF-8')
            return header + dataStr

        logging.info('addThumbnail: making datablock')
        dataBlock = {
            "caption": filename.stem,
            "thumbnail" : encodeImageToThumbnail(filename),
            "datasetId": datasetId,
            "ownerGroup": "BAM 6.5",
        }

        url = self.baseurl + f"{datasetType}/{urllib.parse.quote_plus(datasetId)}/attachments"
        logging.info(f'addThumbnail: sending datablock to url: {url}')
        logging.debug(url)
        ntry=0
        success = False
        while (ntry < self.retries) and not success:
            try:        
                resp = requests.post(
                            url,
                            params={"access_token": self.token},
                            timeout=self.timeouts,
                            stream=False,
                            json = dataBlock,
                            verify=self.sslVerify,
                        )
                success = True
                ntry +=1
            except (OpenSSL.SSL.WantReadError, urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout) as e:
                # sometimes, the server doesn't respond. 
                logging.error(f'server irresponsive, trial {ntry}, trying again if less than {self.retries}')
                ntry +=1
            except: raise    
        return resp
