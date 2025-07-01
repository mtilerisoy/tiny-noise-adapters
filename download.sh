#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Setup
DATASETS_DIR="respiratory_datasets"
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

echo -e "${GREEN}Starting download of respiratory datasets...${NC}"

# 1. Coswara
echo "1. Downloading Coswara..."
git clone --depth 1 https://github.com/iiscleap/Coswara-Data.git || echo "Failed"

# 2. Lung Sounds (Mendeley)
echo "2. Downloading KAUH..."
mkdir -p lung_sounds
cd lung_sounds
wget -c "https://data.mendeley.com/api/datasets/jwyy9np4gv/versions/3/download" -O "data.zip"
unzip -q "data.zip"
cd ..

# 3. Respiratory Patterns (Mendeley)
echo "3. Downloading Respiratory@TR..."
mkdir -p respiratory_patterns 
cd respiratory_patterns
wget -c "https://data.mendeley.com/api/datasets/p9z4h98s6j/versions/1/download" -O "data.zip"
unzip -q "data.zip"
cd ..

# 4. COUGHVID (Zenodo)
echo "4. Downloading COUGHVID..."
mkdir -p coughvid
cd coughvid
wget -c "https://zenodo.org/records/4048312/files/public_dataset.zip?download=1" -O "coughvid.zip"
unzip -q "coughvid.zip"
cd ..

# 5. HF_Lung_V1 (GitLab)
echo "5. Downloading HF_Lung_V1..."
git clone --depth 1 https://gitlab.com/techsupportHF/HF_Lung_V1.git || echo "Failed"

# 6. ICBHI 2017
echo "6. Downloading ICBHI..."
mkdir -p icbhi
cd icbhi
wget --no-check-certificate "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"
unzip -q "ICBHI_final_database.zip"
cd ..


# 8. SPRSound
echo "8. Downloading SPRSound code..."
git clone --depth 1 https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git || echo "Failed"
echo "   Note: Audio data needs separate download from IEEE DataPort"

# 9. CirCor (PhysioNet)
echo "9. Downloading CirCor Heart Sounds..."
mkdir -p circor
cd circor
wget -r -N -c -np "https://physionet.org/files/circor-heart-sound/1.0.3/"
mv physionet.org/files/circor-heart-sound/1.0.3/* .
rm -rf physionet.org
cd ..

echo -e "${GREEN}Done! Check each folder for downloaded data.${NC}"
