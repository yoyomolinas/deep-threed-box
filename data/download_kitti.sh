# Download images. 7481 training, 7518 testing images
if [ !  -d "training/images" ]
then
    if [ ! -f data_object_image_2.zip ]
    then
        echo "Downloading images"
        wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
    fi
    echo "Extracting images"
    unzip data_object_image_2.zip
    mv training/image_2 training/images
    rm data_object_image_2.zip
else 
    echo "Images already exist"
fi

# Donwload calibration files
if [ ! -d "training/calib" ]
then
    if [ ! -f data_object_calib.zip ] 
    then
        echo "Downloading calibration files"
        wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
    fi
    unzip data_object_calib.zip
    rm data_object_calib.zip
else
    echo "Caliobration files already exist"
fi

# Donwload labels
if [ ! -d "training/labels" ]
then
    if [ ! -f data_object_label_2.zip ]
    then
        echo "Downloading labels"
        wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
    fi
    unzip data_object_label_2.zip
    mv training/label_2 training/labels
    rm data_object_label_2.zip
else
    echo "Labels already exist"
fi

# KITTI train/val split used in 3DOP/Mono3D/MV3D
if [ ! -d "training/split" ]
then
    if [ ! -f imagesets.tar.gz ] 
    then
        echo "Downloading KITTI train/val split used in 3DOP/Mono3D/MV3D"
        wget https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
    fi
    tar -xvf imagesets.tar.gz
    mv ImageSets split
    mv split training/split
    rm imagesets.tar.gz
else
    echo "Train/val splits already exist"
fi

# remove testing folder
rm -rf testing