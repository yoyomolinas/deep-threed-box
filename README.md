# Deep 3D Box
This project aims to estimate 3D bounding boxes for pedestrians and vehicles from patches of images, and deploy on embedded devices which will make inferences about the orientation of vehicles and people in public spaces, such as streets and malls. 

### Why?
3D orientation of animate objects around us tell a lot about the activity they're engaged in. It is useful to know the 3D orientation of pedestrians in for example the context of Outdoor Advertising, where orientation can tell a lot about whether a person is exposed to a particular billboard. Similarly, vehicle orientation can tell a lot about the visible range of the people inside. 
 
### How?
The orientation and distance of pedestrians and vehicles are estimated through deep neural networks with a well engineered loss function. 

### Research and Data
This project uses datasets from the following papers:
- [Mono3D++: Monocular 3D Vehicle Detection with Two-Scale 3D Hypotheses and Task Priors](https://arxiv.org/pdf/1901.03446.pdf) 
- [3D Object Proposals for Accurate Object Class Detection](https://proceedings.neurips.cc/paper/2015/file/6da37dd3139aa4d9aa55b8d237ec5d4a-Paper.pdf) 
- [Multi-View 3D Object Detection Network for Autonomous Driving](https://arxiv.org/pdf/1611.07759.pdf) 

### Project Structure
This project extends the [`keras-template`](https://github.com/yoyomolinas/keras-template) project. Please check it out for details.

### Thanks and Contact
This project made possible through many great open source libraries such as Tensorflow, Keras and Opencv, and many great research such as the ones listed above. It is open sourced to support the community of great builders and researchers.

Drop an email at molinas.yoel@gmail.com for questions.
