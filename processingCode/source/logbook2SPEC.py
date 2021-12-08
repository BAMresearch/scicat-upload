# coding: utf-8

# author: Brian R. Pauw, Glen J. Smales
# date: 2019.07.01

# Converter script for generating a SPEC script from a Excel SAXS logbook file.
# ==
#
# New version uses thje SAXSMeasure_BAM_v2 command to collect the data files into a specific path
# new new version uses a new transmission_measure script that's super cool and fancy.

# we need some libraries to run these things.
import pandas, scipy
import numpy as np
import h5py
import sys, os
from io import StringIO
from pathlib import Path
import argparse
from SAXSClasses import readLog


# Here, you define the input excel sheet, and the output script file. These need to exist in the above mentioned directory...
myMeasurementsFile = "logbooks/Logbook_MAUS.xls"

# Some conversion code is defined below. On the last line is the magic line that converts what you have into what you need...

def argparser():
    parser = argparse.ArgumentParser(
        description="""
        A script for converting measurement information in a structured measurement logbook (excel file), 
        into a SPEC script for the MAUS. This SPEC script will ensure all the required measurements are performed.
        This includes the beam profile and center measurements, the incident and transmitted flux, as well as a few other 
        details. 
        
        For best results, combine with Structurize...
        
        example usage:
        python logbook2SPEC_V3.py -i logbooks\Logbook_MAUS.xls -o output_script.spec
        """
    )
    parser.add_argument(
        "-i",
        "--inputFile",
        type=str,
        required=False,
        default=myMeasurementsFile,
        help="Input excel measurement logbook",
    )
    parser.add_argument(
        "-o",
        "--outputFile",
        type=str,
        default='default.mac',
        required=False,
        help="Output filename for the SPEC script",
    )
    return parser.parse_args()

    # You should see a nice, but abbreviated table here with the logbook contents.


def prepSPEC(infile, ofname="default.mac"):
    """ create a SPEC script that can be run. 
    This script should check which measurement filenumbers are already present, and do the missing measurements"""

    rL = readLog(infile)
    L = rL._logbook
    M = L.loc[L.converttoscript].copy()

    # read the additional collimation configurations:
    collPresets = rL._collimation
    prevSource = None  # source in the previous block
    prevConf = None  # configuration in the previous block
    prevBlankYPos = None  # previous BLANKYPOS
    prevBlankZPos = None  # previous BLANKYPOS

    # output file handling:
    # I need an item:
    for index, item in M.iterrows():
        pass 
    # construct the output filename if it has not been explicitly set
    if ofname ==Path('default.mac'): 
        ofname = Path(f'macros','dailies',f'{item.date.year}{item.date.month:02d}{item.date.day:02d}.mac')
    # try to remove if it exists:
    try:
        os.remove(ofname)
    except OSError:
        pass
    

    # script section: for every sample:
    for index, item in M.iterrows():
        sectionscript = []  # put all the SPEC lines in this list

        # change 20210804: moved this out of the repetition section, these settings should be the same for every repetition in a row:
        # find out the source number to use
        source_no = (
            int(np.floor(item.configuration / 100.0)) + 1
        ) % 2 + 1  # 1 = Cu, 2 = Mo
        # set source (1 = Cu, 2 = Mo) if changed
        if (
            prevSource != source_no
        ):  # only add if source has not been defined or is not identical to previous
            configChanged = True # probably not necessary to state this here, but just in case
            sectionscript += ["change_source {}".format(source_no)]
            sectionscript += ["light_off"]
            prevSource = source_no

        # set collimation preset if changed
        if prevConf != item.configuration:
            configChanged = True # so we do a new beam_profile later on
            if not item.configuration in collPresets.index:
                print(
                    "ERROR: Custom configuration {} not found in logbook".format(
                        item.configuration
                    )
                )
                break
            c = collPresets.loc[item.configuration]
            # not moving too much at the same time...
            sectionscript += [
                "umv hp1 {} hp2 {} hp3 {}".format(c.hp1, c.hp2, c.hp3)
            ]
            sectionscript += [
                "umv vp1 {} vp2 {} vp3 {}".format(c.vp1, c.vp2, c.vp3)
            ]
            sectionscript += [
                "umv hg1 {} hg2 {} hg3 {}".format(c.hg1, c.hg2, c.hg3)
            ]
            sectionscript += [
                "umv vg1 {} vg2 {} vg3 {}".format(c.vg1, c.vg2, c.vg3)
            ]
            sectionscript += ["umv detx {} ".format(c.detx)]
            sectionscript += ["umv dety {} ".format(c.dety)]
            sectionscript += ["umv detz {} ".format(c.detz)]
            # move beamstop back
            sectionscript += ["umv bsz {} bsr {} ".format(c.bsz, c.bsr)]
            # finished configuring
            prevConf = item.configuration

        # set BLANKYPOS when changed:
        if prevBlankYPos != item.blankpositiony:
            sectionscript += ["BLANKYPOS={}".format(item.blankpositiony)]
            prevBlankYPos = item.blankpositiony

        # set BLANKZPOS when changed:
        if prevBlankZPos != item.blankpositionz:
            sectionscript += ["BLANKZPOS={}".format(item.blankpositionz)]
            prevBlankZPos = item.blankpositionz

        # end change 20210804


        for repetition in range(item.nrep):
            configChanged = False
            # debug
            print(
                "# preparing entry in row: {}, filenum:{}".format(
                    index, item.filenum + repetition
                )
            )
            # for info in the SPEC script
            sectionscript += [
                "# preparing entry in row: {}, filenum:{}".format(
                    index, item.filenum + repetition
                )
            ]


            # change 20201222: always beam profile through transmission_measure_bam
            ODBName = '"/home/saxslab/data/{year}/{year}{month:02d}{day:02d}/{year}{month:02d}{day:02d}_{ofn}/beam_profile"'.format(
                year=item.date.year,
                month=item.date.month,
                day=item.date.day,
                ofn=(item.filenum + repetition),
            )
            ODBSName = '"/home/saxslab/data/{year}/{year}{month:02d}{day:02d}/{year}{month:02d}{day:02d}_{ofn}/beam_profile_through_sample"'.format(
                year=item.date.year,
                month=item.date.month,
                day=item.date.day,
                ofn=(item.filenum + repetition),
            )

            # here we split: are we doing USAXS measurements or not?
            if not item.usaxs:

                
                # umv to sample position
                sectionscript += [
                    "umv ysam {} zsam {}".format(item.positiony, item.positionz)
                ]

                # move beamstop out for measuring a beam profile
                # set sample description
                sectionscript += [
                    'SAMPLE_DESCRIPTION="beam_profile for {}"'.format(item.samplename)
                ]  # name of the sample
                # set FRAME_TIME
                sectionscript += ["FRAME_TIME={}".format(0.2)]

                # fancy transmission_measure
                sectionscript += [f"transmission_measure_bam {ODBName} {ODBSName}"]
                # set beamstop in or out
                if item.beamstop:
                    sectionscript += ["umv bsz {} bsr {} ".format(c.bsz, c.bsr)]
                else:
                    sectionscript += ["bstop_out"]

                # set output directory
                ODName = '"/home/saxslab/data/{year}/{year}{month:02d}{day:02d}/{year}{month:02d}{day:02d}_{ofn}"'.format(
                    year=item.date.year,
                    month=item.date.month,
                    day=item.date.day,
                    ofn=(item.filenum + repetition),
                )
                # ensure output directory exists
                # sectionscript += ['unix( "mkdir -p {OD}")'
                #                   .format(OD = ODName)]

                # set sample description
                sectionscript += [
                    'SAMPLE_DESCRIPTION="{}"'.format(item.samplename)
                ]  # name of the sample
                # set thickness
                sectionscript += [
                    "SAMPLE_THICKNESS={}".format(item.samplethickness * 100.0)
                ]  # specified in cm, for some reason.
                # set FRAME_TIME
                sectionscript += ["FRAME_TIME={}".format(item.frametime)]
                # saxsmeasure [duration]
                sectionscript += ["saxsmeasure_bam {} {}".format(item.duration, ODName)]

            if item.usaxs:
                # each repetition should be a scan, 100 points for each side-band, 64 points for the beam profile. Each of these should be a measurement. 


                # umv to sample position
                sectionscript += [
                    "umv yheavy {} zheavy {}".format(item.positiony, item.positionz)
                ]
                # first 100 points, from -2.5 mrad (q at moly energy about 0.05) to -0.08 mrad, log-spaced
                hangles = []
                # 100 points
                hangles += [np.geomspace(-1.4, -0.08, 100, endpoint = False)]
                # 80 points
                hangles += [np.arange(-0.08, 0.08, 0.002)]
                # 100 points
                hangles += [np.geomspace(0.08, 1.4, 100, endpoint = True)]
                hangles = np.concatenate(hangles)

                htimeFactor = abs(hangles / hangles.max() * 2).clip(0.04,1) # reduce measurement times from 100% to 4% as we're getting closer to the beam
                htimes = np.round(item.duration * htimeFactor) # exposure times
                hftimes = htimes / 10
                print(f' * * * * total time for this scan: {htimes.sum() / 3600} h * * * * ')

                for hindex, hangle in enumerate(hangles):
                    # move fineyaw to target
                    sectionscript += [
                        f'umv fineyaw {hangle}'
                    ] 

                    # set sample description
                    sectionscript += [
                        f'SAMPLE_DESCRIPTION="{item.samplename} at fineyaw = {hangle} mrad"'
                    ]  # name of the sample
                    # set thickness.. are we even still using this value? I think it's modded post-meas.
                    sectionscript += [
                        "SAMPLE_THICKNESS={}".format(item.samplethickness * 100.0)
                    ]  # specified in cm, for some reason.

                    # set FRAME_TIME
                    sectionscript += ["FRAME_TIME={}".format(hftimes[hindex])]
                    
                    # beamstop needs to be out of the way for these measurements
                    sectionscript += ["bstop_out"]
                    # set output directory
                    ODName = '"/home/saxslab/data/{year}/{year}{month:02d}{day:02d}/{year}{month:02d}{day:02d}_{ofn}/USAXS_{hidx}"'.format(
                        year=item.date.year,
                        month=item.date.month,
                        day=item.date.day,
                        ofn=(item.filenum + repetition),
                        hidx=hindex,
                    )
                    # saxsmeasure [duration]
                    sectionscript += ["saxsmeasure_bam {} {}".format(htimes[hindex], ODName)]

                pass
        # finally, transfer data at the end of the script

        sectionscript += ['unix("./syncdata.sh")']
        print(f'writing to ofname {ofname}')
        with open(ofname, "at", encoding="utf-8") as myfile:
            myfile.write("\n".join(sectionscript))
            myfile.write("\n\n")


# Run this magic line to create the output script..
if __name__ == "__main__":
    adict = argparser()
    adict = vars(adict)
    prepSPEC(Path(adict["inputFile"]), Path(adict["outputFile"]))
