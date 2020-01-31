# Example usage :
# bash to_edgetpu.sh --keras_model agegender.hdf5 --out edgetpu/

while [ "$1" != "" ]; do
    case $1 in
        -mdl | --keras_model )         shift
                                            KERAS_MODEL=$1;;
        -out | --out )     OUTPUT_DIRECTORY=$2;;
        
    esac
    shift
done

source activate tf2 # Must use environment with tf-nightly 
echo $KERAS_MODEL
python to_tflite.py --keras_model $KERAS_MODEL --out agegender.tflite

mkdir -p $OUTPUT_DIRECTORY
echo "Compiling for Edge TPU."
edgetpu_compiler agegender.tflite --out_dir $OUTPUT_DIRECTORY

# echo "Cleaning up."
# rm agegender.tflite