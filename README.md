# cnn-iris-recognition-framework

## Introduction
This is a framework built around PyTorch for configuring research experiments for iris recognition research using convolutional neural networks. The framework provides a streamlined way of configuring datasets, models, and model training. The framework also provides ways of producing different model evaluations including accuracy, accuracy roc, and computing Euclidean distances.

### Prexisiting Configurations
The framework comes with pre-existing configurations for the following CNNs:
- AlexNet
- ResNet50
- ResNet101
- ResNet152

The framework also has dataset configuration files for configuring training, testing, and validating a model for the effects of blinking on iris recognition as well as a script for each of these operations. There are also existing flags for the dataset cohorts and images. The dataset is not available in this repository. However, some sample images can be found in [docs/sample-dataset]()

## Building the Framework
The framework uses multiple external and Python dependencies. The dependency installation instructions can be found [here](/docs/dependencies.md).

## Sample Results
The following results were obtained using the ResNet50 CNN and a dataset created to test the effects of blinking on iris recognition. The specific sample results shown are graphs for model weights obtained by training using a dataset with an original iris occlusion distribution (with most eyes being more than 80% open) plus a dataset of synthetically occluded iris images, with cohorts ranging from 0% occlusion to 60% occlusion. The validation dataset was split into 10 cohorts, each with a different level of occlusion (ranging from 0 to 90%). A sample of the dataset can be seen [here](/docs/sample-dataset/). The following graphs show the results of such an experiment:

<p float="left">
    <img src="/assets/fig/resnet50_original_plus_0_to_60_percent_synthetic_occlusion/accuracy_bargraph.png" width="500"/>
    <img src="/assets/fig/resnet50_original_plus_0_to_60_percent_synthetic_occlusion/accuracy_roc.png" width="500"/>
</p>

<div style="display: flex; flex-wrap: wrap;">
    <img src="/assets/fig/resnet50_original_plus_0_to_60_percent_synthetic_occlusion/0_percent_occlusion_dataset_euclideandist_histogram.png" width="500"/>
    <img src="/assets/fig/resnet50_original_plus_0_to_60_percent_synthetic_occlusion/30_percent_occlusion_dataset_euclideandist_histogram.png" width="500"/>
    <img src="/assets/fig/resnet50_original_plus_0_to_60_percent_synthetic_occlusion/60_percent_occlusion_dataset_euclideandist_histogram.png" width="500"/>
    <img src="/assets/fig/resnet50_original_plus_0_to_60_percent_synthetic_occlusion/90_percent_occlusion_dataset_euclideandist_histogram.png" width="500"/>
</div>

A presentation of the results for the whole experiment can be found [here](/assets/effect-of-blink-on-iris-presentation.pptx) or [downloaded](https://github.com/danieltebor/cnn-iris-recognition-framework/raw/main/docs/effect-of-blink-on-iris-presentation.pptx)).
