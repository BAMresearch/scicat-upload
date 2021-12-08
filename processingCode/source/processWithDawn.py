# coding: utf-8

# author: Brian R. Pauw

import pandas
from pathlib import Path
import numpy as np
import json
import argparse
from SAXSClasses import readLog  # reads our logbooks
import multiprocessing
import subprocess

# DAWNCommand = "Y:\Software\SAXS002\DawnDiamond-2.18.0.v20200527-1217-windows64\dawn.exe"
# DAWNCommand = '/opt/dawn/DawnDiamond-2.18.0.v20200611-2026-linux64/dawn'
DAWNCommand = '/opt/dawn/DawnDiamond-2.23.0.v20211005-0856/dawn'
# DAWNCommand = '/opt/dawn/DawnDiamond-2.25.0.v20211108-1510/dawn'

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
    return parser.parse_args()

def runRow(item):
    # structurizes the files from a logbook entry row
    
    ifname = Path(
        f"data/{item.date.year}/{item.YMD}/{item.YMD}_{item.filenum}_expanded_stacked.nxs"
    )
    assert ifname.exists(), f"input file {ifname} does not exist"
    ofname = Path(
        f"data/{item.date.year}/{item.YMD}/autoproc/group_{item.ofgroup}/{item.YMD}_{item.filenum}_expanded_stacked_processed.nxs"
    )
    # ensure autoproc path exists:
    ofname.parent.mkdir(parents=True, exist_ok=True)
    if ofname.exists():
        ofname.unlink() # remove processed file if exists.
    ojfname = Path(
        f"data/{item.date.year}/{item.YMD}/autoproc/{item.YMD}_{item.filenum}_autoDAWNSettings.json"
    )

    oDict = {
        "runDirectory": Path(f"data/{item.date.year}/{item.YMD}").resolve().as_posix(),
        "name": "TestRun",
        "filePath": ifname.resolve().as_posix(),
        "dataDimensions": [-1, -2],
        "processingPath": Path(f"data/Pipelines/{item.procpipeline}").resolve().as_posix(),
        "outputFilePath": ofname.resolve().as_posix(),
        "deleteProcessingFile": False,
        "datasetPath": "/entry1/instrument/detector00/data",
        "numberOfCores" : 1, 
        "xmx" : 2048
    }

    with open(ojfname, 'w') as json_file:
        json.dump(oDict, json_file, indent = 4)    

    # run DAWN on mac:
    subprocess.call([
        DAWNCommand,
        '-noSplash', 
        '-configuration', (Path.home()/'.dawn').resolve().as_posix(),
        '-application', 'org.dawnsci.commandserver.processing.processing',
        '-path', ojfname.resolve().as_posix()
        ])

if __name__ == "__main__":
    adict = argparser()
    adict = vars(adict)

    myMeasurementsFile = Path(adict["inputFile"])

    baseMonth = adict["baseDate"]
    year = baseMonth[0:4]
        # do the nexus conversion of every individual file, to a one-frame NeXus image.
    logInstance = readLog(myMeasurementsFile)
    lbEntry = logInstance.getMainEntry(YMD = baseMonth)
    ## single threaded:
    # for index, item in lbEntry.iterrows():
    #     runRow(item)
    ## multithreaded:
    Pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    # Pool = multiprocessing.Pool(processes = 1)
    mapParam = [item for index, item in lbEntry.iterrows()]
    rawData = Pool.map(runRow, mapParam)    
    Pool.close()
    Pool.join()
