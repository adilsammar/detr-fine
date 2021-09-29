![rmc_batching_plant](./assets/panoptic/rmc_batching_plant_5155.png)

# Train DETR for Panoptic Segmentation

Computer vision and scene understanding have become game-changer in today’s world. As we move forward into giving autonomous capabilities to machines to perform tasks in a human-way fashion, understanding the surroundings, objects around, and scenes becomes pivotal. 

Panoptic segmentation combines instance segmentation and semantic segmentation to provide a more holistic understanding of a given scene than the latter two alone. In this post, I will walk you through the concept of panoptic segmentation and how it is helping machines to view the world the way we see it.


## What is Segmentation
### Semantic Segmentation

Semantic segmentation refers to the task of classifying pixels in an image. It is done by predefining some target classes, e.g., “car”, “vegetation”, “road”, “sky”, “sidewalk”, or “background”, where “background” is in most cases a default class. Then, each pixel in the image is assigned to one of those classes. Here’s an example:

![Input](./assets/input.png)

![Output](./assets/output.png)

As you can see in the previous example, every pixel in the image was colored depending on its class; hence, every pixel belonging to a car is masked in blue and the same goes for the sidewalk, the vegetation, road, and the sky.

if we want to dig deeper into the type of information we can extract here. Say, for example, we want to know how many cars are in one picture. Semantic segmentation is of no help here as all we can get is a pixel-wise classification. For such a task, we need to introduce the concept of object detection and instance segmentation.

### Object Detection and Instance Segmentation

When we do object detection, we aim to identify bounded regions of interest within the image inside of which is an object. Such objects are countable things such as cars, people, pets, etc. It doesn’t apply to classes such as “sky” or “vegetation” since they are usually spread in different regions of the image, and you cannot count them one by one since there’s only one instance of them — there is only one “sky” not multiple.

It is very common to use bounding boxes to indicate the region within which we will find a given object. Here’s an example:

![Boxes](./assets/boxes.png)

In the previous image, there are three bounding boxes, one for each car on the image. In other words, we are detecting cars, and we can now say how many of them are in the image.

Now, not all the pixels inside those bounding boxes correspond to a car. Some of those pixels are part of the road; others of the sidewalk or the vegetation. If we want to obtain richer information from object detection, we can identify what pixels specifically belong to the same class assigned to the bounding box. That is what is called instance segmentation. Strictly speaking, we perform pixel-wise segmentation for every instance (bounding box in our case) we detected. This is what it looks like:

![Instance](./assets/instance.png)

So we went from a rough detection with a bounding box to a more accurate detection in which we can also identify instances and therefore count the number of objects of a given class. In addition to that, we know exactly what pixels belong to an object.

Still we have no information about all the other non-instance classes such as “road”, “vegetation” or “sidewalk” as we did have it in semantic segmentation. That is when panoptic segmentation comes into play!

### Panoptic Segmentation

As mentioned in the introduction of this post, panoptic segmentation is a combination of semantic segmentation and instance segmentation. To put it another way , with panoptic segmentation, we can obtain information such as the number of objects for every instance class (countable objects), bounding boxes, instance segmentation. But, also we get to know what class every pixel in the image belongs to using semantic segmentation. This certainly provides a more holistic understanding of a scene.

Following our example, panoptic segmentation would look like this:

![Panoptic](./assets/panoptic.png)

We have now managed to get a representation of the original image in such a way that it provides rich information about both semantic and instance classes altogether.



## Train DETR

Training DETR is a two step process

1. First train DETR for Object detection
2. Then add panoptic mask head and freeze base network and further train for a couple of epochs


In our [previous article](README.md) we have shown how to create a custom dataset from scratch and train Object Detection Model on the same. We have trained object detection model in multiple steps for around 150 epochs.

In this step we are adding Panoptic head on top of base DETR model and train it for another 150 epochs. But in this part we will train panoptic head with base model for first 500 epochs then we are goiong to freeze object detection model and train further for 50 epochs.


**Step 1:** Create Data Loader at [detr/datasets/construction_panoptic.py](./detr/datasets/construction_panoptic.py)





## Metrics

```
--------------------------------------
All       |  46.2   71.1   60.0    59
Things    |  52.5   72.4   67.3    44
Stuff     |  27.9   67.4   38.4    15

```


## References

* [Panoptic Segmentation Explained](https://hasty.ai/blog/panoptic-segmentation-explained)
* [END-to-END object detection (Facebook AI)](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers)
* [Attention is All you Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
* [The Annotated DETR](https://amaarora.github.io/2021/07/26/annotateddetr.html)
* [Facebook DETR: Transformers dive into the Object Detection World](https://towardsdatascience.com/facebook-detr-transformers-dive-into-the-object-detection-world-39d8422b53fa)