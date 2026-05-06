# Cloud Chaser Academic Pipeline

## Concept

Cloud Chaser is a two-stage vision system for cloud identification. The first stage segments cloud pixels from sky imagery using YOLO-Seg trained on SWIMSEG/SkyImage binary masks. The second stage classifies each detected cloud crop into a meteorological category using a CNN trained on GCD.

Given an image \(I\), the system predicts cloud instances:

\[
\mathcal{P} = \{(M_i, B_i, c_i, s_i, p_i)\}_{i=1}^{N}
\]

where \(M_i\) is a mask, \(B_i\) is a box, \(c_i\) is the class label, \(s_i\) is detector confidence, and \(p_i\) is classifier confidence.

## Datasets

### Segmentation: SWIMSEG / SkyImage

SWIMSEG provides explicit cloud masks and is the primary detector dataset. The local converter discovers image-mask pairs, binarizes masks, extracts connected components, converts components to YOLO polygons, and writes a one-class cloud segmentation dataset.

### Classification: GCD

GCD supplies seven cloud-type labels: cumulus, altocumulus, cirrus, clear sky, stratocumulus, cumulonimbus, and mixed. The local loader supports both `train/<class>/*.jpg` and direct `<class>/*.jpg` layouts.

### Optional SSL: TJNU

TJNU can be used for SimCLR pretraining. If TJNU is absent, the local training command skips SSL and the classifier starts from ImageNet-pretrained weights.

## Pipeline

1. Convert SWIMSEG masks to YOLO-Seg labels.
2. Fine-tune `yolo11s-seg.pt` on cloud masks.
3. Optionally pretrain the classifier encoder with SimCLR on TJNU.
4. Fine-tune a ResNet50/EfficientNet/DenseNet classifier on GCD.
5. During inference, detect cloud masks, crop masked regions, classify crops, and overlay labels.

## Metrics

Segmentation is evaluated with mask mAP, mask mAP@50, and foreground mIoU on the SWIMSEG validation split. Classification is evaluated with Top-1 accuracy and macro F1-score on GCD.

## Limitations

SWIMSEG improves cloud-mask supervision over generic scene datasets, but it is mostly sky-image oriented. General outdoor scenes with trees, buildings, and mountains may require additional domain data for maximum robustness.
