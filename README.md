# Estimation of 3D bounding boxes

Sample camera matrix: 


    [[7.215377e+02, 0.          , 6.095593e+02, 4.485728e+01],
     [0.          , 7.215377e+02, 1.728540e+02, 2.163791e-01],
     [0.          , 0.          , 1.          , 2.745884e-03]]

To Do : 
1. Split dataset using trainval split in used in 3DOP/Mono3D/MV3D"
2. Train in new instance
3. Deployment script - End to end deployment given model name and path to save - should include keras, tflite, edgetpu model files
4. Recovery of angles in deployed model
5. Efficient net impl
