# End-to-End Training DETR on custom DATA

In this article we will cover training Detr on custom dataset end to end. Following are the points we will cover over multiple documents

Before diving into theory of how it is done lets get insipired with this video

[![Construction Panoptic](https://img.youtube.com/vi/7mDA3SvYjiw/0.jpg)](https://www.youtube.com/watch?v=7mDA3SvYjiw)

1. [COCO Format for Object detection](./OBJECTDETECTION.md#COCOFormatforObjectdetection)
	* 1.1. [COCO file format](./OBJECTDETECTION.md#COCOfileformat)
	* 1.2. [Folder Structure](./OBJECTDETECTION.md#FolderStructure)
	* 1.3. [JSON format](./OBJECTDETECTION.md#JSONformat)
2. [Creating a Custom COCO format dataset](./OBJECTDETECTION.md#CreatingaCustomCOCOformatdataset)
	* 2.1. [Background](./OBJECTDETECTION.md#Background)
	* 2.2. [Add Masks for Stuff Classes](./OBJECTDETECTION.md#AddMasksforStuffClasses)
	* 2.3. [Example Mask Image](./OBJECTDETECTION.md#ExampleMaskImage)
3. [DETR in depth](./DETREXPLAINED.md#detr-in-depth)
4. [Fine Tune DETR on custom dataset for Object Detection](./OBJECTDETECTION.md#FineTuneDETRoncustomdatasetforObjectDetection)
	* 4.1. [Prepare Code](./OBJECTDETECTION.md#PrepareCode)
	* 4.2. [Train Model](./OBJECTDETECTION.md#TrainModel)

