# End-to-End Training DETR on custom DATA

In this article we will cover training Detr on custom dataset end to end. Following are the points we will cover over multiple documents

Before diving into theory of how it is done lets get insipired with this video

[![Construction Panoptic](https://img.youtube.com/vi/7mDA3SvYjiw/0.jpg)](https://www.youtube.com/watch?v=7mDA3SvYjiw)

1. [Object Detection Background](./OBJECTDETECTION.md#train-detr-for-object-detection-on-custom-data)
2. [What is Segmentation](#WhatisSegmentation)
	* 2.1. [Semantic Segmentation](#SemanticSegmentation)
	* 2.2. [Object Detection and Instance Segmentation](#ObjectDetectionandInstanceSegmentation)
	* 2.3. [Panoptic Segmentation](#PanopticSegmentation)
3. [COCO Format for Object detection](./OBJECTDETECTION.md#COCOFormatforObjectdetection)
	* 3.1. [COCO file format](./OBJECTDETECTION.md#COCOfileformat)
	* 3.2. [Folder Structure](./OBJECTDETECTION.md#FolderStructure)
	* 3.3. [JSON format](./OBJECTDETECTION.md#JSONformat)
4. [Creating a Custom COCO format dataset](./OBJECTDETECTION.md#CreatingaCustomCOCOformatdataset)
	* 4.1. [Background](./OBJECTDETECTION.md#Background)
	* 4.2. [Add Masks for Stuff Classes](./OBJECTDETECTION.md#AddMasksforStuffClasses)
	* 4.3. [Example Mask Image](./OBJECTDETECTION.md#ExampleMaskImage)
5. [DETR in depth](./DETREXPLAINED.md#detr-in-depth)
6. [Fine Tune DETR on custom dataset for Object Detection](./OBJECTDETECTION.md#FineTuneDETRoncustomdatasetforObjectDetection)
	* 6.1. [Prepare Code](./OBJECTDETECTION.md#PrepareCode)
	* 6.2. [Train Model](./OBJECTDETECTION.md#TrainModel)
7. [Example Predictions](./OBJECTDETECTION.md#ExamplePredictions)
8. [Train DETR for Panoptic Segmentation](SEGMENTATION.md#TrainDETR)
	* 8.1. [Prepare Code](SEGMENTATION.md#Steps)
	* 8.2. [Train model with panoptic head](SEGMENTATION.md#Trainmodelwithpanoptichead)
8. [Example Predictions](SEGMENTATION.md#ExamplePredictions)

