#!/bin/bash
echo "We are about to:"
echo " - download the EuRoC machine hall 1 dataset,"
echo " - extract the 752x480 images from the bag file, and"
echo " - also crop the images to 640x480"
echo "Make sure you have enough space on your disk ~ 4 GB!"
echo ""
read -p "[0] Are you sure you want to do this? [y/n] " ANSWER
if [ "${ANSWER}" != "y" ]; then
  exit 0
fi

TARGET_FOLDER="euroc"
BAG_NAME="MH_01_easy.bag"


echo "[1] Dataset ready"
read -p " Extract all images? [0 = all / N] " IMAGE_CNT_LIMIT
echo " Next: start extraction"
sleep 1

# At this point we have either a symlink or file in the folder
# Extract the images
EXTRACT_PYTHON_SCRIPT_NAME="extract_python.py"

echo "[2] Extraction is ready"
echo " Next: start cropping to 640x480"
sleep 1

# do the cropping
DO_CROPPING=1
if [ ${DO_CROPPING} -eq 1 ]; then
  mkdir -p images/640_480
  cd images/752_480
  for file in *.png; do
    convert -crop 640x480+56+0 $file ../640_480/${file%.png}.png
    echo ${file}
  done
else
  cd images/752_480
fi

IMAGE_COUNT=$(ls -1 -v | wc -l)

echo "[3] Images are ready"
echo " Next: create list files for the test environment"
echo " Image count: "${IMAGE_COUNT}
sleep 1

cd ../..

IMAGE_LIST_FILE_752_480="image_list_752_480.txt"
IMAGE_LIST_FILE_640_480="image_list_640_480.txt"

echo -e "752\n480" > ${IMAGE_LIST_FILE_752_480}
ls -v -1 "images/752_480" | awk -v path="test/images/${TARGET_FOLDER}/images/752_480" '{print ""path"/"$1""}' >> ${IMAGE_LIST_FILE_752_480}
echo -e "640\n480" > ${IMAGE_LIST_FILE_640_480}
ls -v -1 "images/752_480" | awk -v path="test/images/${TARGET_FOLDER}/images/640_480" '{print ""path"/"$1""}' >> ${IMAGE_LIST_FILE_640_480}

echo "DONE"
