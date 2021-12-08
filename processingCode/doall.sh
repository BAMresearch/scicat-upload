#!/bin/bash
# usage: doall.sh [basedate]
echo "working on basedate: $1"
cd /mnt/vsi-db/Measurements/SAXS002/

# structurize:
python processingCode/source/Structurize.py -i logbooks/The_Logbook_MAUS.xls -b $1

baseyear=`echo $1 | awk '{print substr($0,0,4)}'`
baseyear_Proposals=`echo $1 | awk '{print substr($0,0,4)}'`
# upload:
toProcess="/mnt/vsi-db/Measurements/SAXS002/data/$baseyear/$1/"
monitorFiles="$baseyear????_*_expanded_stacked.nxs"

# start with uploading the relevant proposals and samples:
python processingCode/source/proposalUploadByYMD.py -i logbooks/The_Logbook_MAUS.xls --proposalPath "../../Proposals/SAXS002/$baseyear_Proposals" --YMD "$1"	# >> ~/raw.log 2&>1

for f in `find "$toProcess" -maxdepth 1 -name "$monitorFiles" -type f`
do
        echo "$f" # >> ~/raw.log 2&>1
		python processingCode/source/autoThickness.py "$f"
        python processingCode/source/genSciCatThumbnail.py -t raw -f "$f" # >> ~/raw.log 2&>1
        python processingCode/source/datasetUpload.py -i logbooks/The_Logbook_MAUS.xls -u archiveManager -p FIXME -t raw -f "$f" &	# >> ~/raw.log 2&>1
done

# process with Dawn:
python processingCode/source/processWithDawn.py -i logbooks/The_Logbook_MAUS.xls -b $1

# upload processed:
monitorFiles="$baseyear????_*_expanded_stacked_processed*.nxs"
for f in `find "$toProcess" -maxdepth 4 -name "$monitorFiles" -type f`
do
        echo "$f" # >> ~/derived.log 2&>1
        python processingCode/source/genSciCatThumbnail.py -t derived -f "$f" # >> ~/raw.log 2&>1
        python processingCode/source/datasetUpload.py -i logbooks/The_Logbook_MAUS.xls -u archiveManager -p FIXME -t derived -f "$f" & # >> ~/derived.log 2&>1
done

