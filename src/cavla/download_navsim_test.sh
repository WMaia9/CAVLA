#!/bin/bash
# NAVSIM navtest inputs (OpenScene v1.1 test split) and nuPlan maps, following navsim/download/download_test.sh
# and download_maps.sh. Resumable: finished archives are kept as .done markers and skipped.
set -e
cd /mnt/datasets/CAVLA/data/navsim
fetch() {  # url, then extract the archive and mark it done
  name=$(basename "$1")
  [ -f "$name.done" ] && return 0
  wget -q -c "$1" -O "$name"
  case "$name" in *.tgz) tar -xzf "$name";; *.zip) unzip -q -o "$name";; esac
  rm -f "$name"; touch "$name.done"; echo "done $name"
}
fetch https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
fetch https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz
for split in $(seq 0 31); do
  fetch https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_${split}.tgz
  fetch https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_lidar/openscene_sensor_test_lidar_${split}.tgz
done
echo ALL_DONE
