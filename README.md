# Cloud Chaser

## Abstract

Cloud Chaser is a two-stage computer vision system for cloud identification in unconstrained outdoor imagery. The system first localizes cloud-like regions using an instance segmentation model trained on environmental scene annotations, then classifies each detected cloud region into a meteorological category using a deep convolutional classifier. The design separates geometric localization from cloud-type recognition, allowing the segmentation module to handle background clutter such as buildings, trees, mountains, and blue sky, while the classification module focuses on texture, shape, and spatial structure within extracted cloud regions.

## 1. Problem Definition

The objective is to process a raw outdoor image and produce pixel-level cloud masks, bounding boxes, confidence scores, and meteorological cloud-type labels. Formally, given an image \(I \in \mathbb{R}^{H \times W \times 3}\), the system predicts a set of cloud instances:

\[
\mathcal{P} = \{(M_i, B_i, c_i, s_i, p_i)\}_{i=1}^{N}
\]

where \(M_i\) is a binary segmentation mask, \(B_i\) is a bounding box, \(c_i\) is the predicted cloud type, \(s_i\) is the segmentation confidence, and \(p_i\) is the classification confidence.

The task is challenging because clouds have ambiguous boundaries, variable scale, high intra-class diversity, and frequent visual overlap with sky, haze, smoke, snow, bright buildings, and other outdoor structures.

## 2. Dataset Strategy

### 2.1 Localization Dataset: ADE20K

ADE20K is used to train the segmentation component because it contains general-scene semantic annotations. Rather than training only on isolated sky imagery, ADE20K exposes the model to realistic environmental context, including urban, rural, coastal, mountainous, and indoor/outdoor transition scenes.

The pipeline extracts the relevant foreground classes, primarily `sky` and, when available, `cloud`. In this repository’s ADE20K metadata, `sky` is available as class ID `3`, while `cloud` is not present in the 150-class mapping. Therefore, the converter is configurable and falls back to `sky` when a dedicated cloud label is unavailable. This produces a one-class YOLO segmentation dataset where the foreground class is treated as cloud/sky-region candidate material.

### 2.2 Classification Dataset: GCD

The GCD dataset provides labeled images for seven cloud categories:

1. Cumulus
2. Altocumulus
3. Cirrus
4. Clear sky
5. Stratocumulus
6. Cumulonimbus
7. Mixed cloud

The classifier is trained using standard train, validation, and test partitions. If no validation directory exists, a stratified validation subset is deterministically sampled from the training set.

### 2.3 Self-Supervised Dataset: TJNU

The unlabeled TJNU dataset is used for self-supervised representation learning. Since cloud morphology is strongly defined by texture, density, illumination, and spatial structure, self-supervised pretraining helps the encoder learn useful visual embeddings before supervised fine-tuning on the smaller labeled GCD set.

## 3. Data Preprocessing and Augmentation

### 3.1 ADE20K Mask Conversion

ADE20K semantic masks are converted into YOLO segmentation labels. For each ADE mask, pixels belonging to the selected class IDs are combined into a binary foreground mask. Connected components are extracted, filtered by minimum area, converted into polygon contours, normalized to image coordinates, and written in YOLO segmentation format.

This conversion enables fine-tuning YOLO-Seg models using pixel-level supervision while preserving instance-like connected regions.

### 3.2 Meteorology-Aware Augmentations

The classification and self-supervised pipelines use augmentations selected to preserve meteorological texture patterns while improving robustness:

- Random shadows simulate illumination changes and partial occlusions.
- Gaussian blur models atmospheric haze, lens softness, and motion blur.
- Horizontal flips preserve cloud texture statistics and increase viewpoint diversity.
- Conservative vertical flips are included because cloud texture is often orientation-tolerant, but the probability is kept lower than horizontal flips to avoid excessive physical distortion.
- Color jitter accounts for exposure, white balance, and time-of-day variation.

## 4. Segmentation Module

### 4.1 Architecture

The segmentation module uses an Ultralytics YOLO segmentation architecture, configured by default as `yolo11s-seg.pt`. The model may also be switched to `yolov8s-seg.pt` through the YAML configuration.

YOLO-Seg is used because it provides:

- real-time or near-real-time detection speed,
- bounding boxes and masks from a single model,
- strong transfer learning from pretrained weights,
- straightforward export to ONNX and other deployment formats.

### 4.2 Training Objective

The detector is fine-tuned on the converted ADE20K masks. Its objective is to distinguish cloud/sky foreground from non-cloud background objects such as buildings, vegetation, roads, and mountains. The output consists of binary instance masks, bounding boxes, and confidence scores.

### 4.3 Output

For each detected instance, the detector returns:

- a bounding box \(B_i = (x_1, y_1, x_2, y_2)\),
- a pixel mask \(M_i\),
- a detection confidence \(s_i\).

These outputs are passed to the classification stage.

## 5. Feature Extraction and Classification Module

### 5.1 Two-Stage Recognition

The system uses a two-stage recognition strategy:

1. Localize candidate cloud regions with YOLO-Seg.
2. Classify each detected cloud crop using a CNN classifier.

This design is preferred over whole-image classification because a general outdoor image may contain many irrelevant objects. Mask-guided cropping reduces background bias and encourages the classifier to focus on cloud morphology.

### 5.2 Backbone Choices

The classifier supports the following convolutional backbones:

- ResNet50
- EfficientNet-B0
- DenseNet121

ResNet50 is the default because it offers a strong balance between representational capacity, training stability, and deployment compatibility.

### 5.3 Self-Supervised Pretraining

The implemented self-supervised method is SimCLR. Given an unlabeled cloud image, two augmented views are generated and passed through a shared encoder and projection head. The NT-Xent contrastive loss pulls embeddings from the same image together while pushing embeddings from different images apart.

This produces a cloud-aware encoder that can learn visual cues such as:

- fibrous cirrus texture,
- dense cumulonimbus vertical structure,
- stratocumulus layering,
- isolated cumulus contours,
- mixed cloud heterogeneity.

After pretraining, the encoder weights are transferred into the supervised classifier and fine-tuned on GCD labels.

### 5.4 Supervised Fine-Tuning

The classification head is trained with cross-entropy loss over the seven GCD classes. The training pipeline supports optional backbone freezing for early epochs, which stabilizes fine-tuning when the classifier is initialized from self-supervised weights.

## 6. Inference Pipeline

At inference time, the system processes a high-resolution image through the following steps:

1. Read the raw RGB image.
2. Run YOLO-Seg to detect cloud instances.
3. Resize masks to the original image resolution when necessary.
4. Use each mask and bounding box to extract a cloud-focused crop.
5. Batch all crops and pass them through the classifier.
6. Convert logits into class probabilities using softmax.
7. Overlay masks, bounding boxes, class names, and confidence scores on the original image.

The final output is an annotated image and a structured list of predictions.

## 7. Optimization and Deployment

The implementation supports several production-oriented optimizations:

- mixed precision training and inference on CUDA,
- batched classification of detected crops,
- configurable YOLO confidence and IoU thresholds,
- ONNX export for detector and classifier,
- TorchScript export for classifier deployment,
- centralized YAML configuration for reproducible experiments.

The modular code structure separates data processing, model definitions, training logic, evaluation, inference, visualization, and export.

## 8. Evaluation Metrics

### 8.1 Segmentation Metrics

The segmentation module is evaluated using:

- mask mAP, reported by Ultralytics validation,
- mask mAP@50,
- mean Intersection over Union over ADE20K validation foreground masks.

For binary mIoU, the predicted foreground mask is compared with the target cloud/sky mask:

\[
\text{IoU} = \frac{|M_{pred} \cap M_{gt}|}{|M_{pred} \cup M_{gt}|}
\]

### 8.2 Classification Metrics

The classifier is evaluated using:

- Top-1 accuracy,
- macro F1-score.

Macro F1 is important because cloud datasets may be class-imbalanced, and average accuracy alone can obscure weak performance on rarer cloud types such as cumulonimbus.

## 9. Reproducibility

The pipeline uses deterministic train/validation splitting for GCD and a centralized configuration file. Important experimental parameters are stored in `configs/default.yaml`, including dataset roots, image size, model backbones, learning rates, batch sizes, epoch counts, augmentation probabilities, detector thresholds, and checkpoint paths.

## 10. Limitations

The current ADE20K segmentation supervision depends on available semantic classes. If a dataset provides explicit cloud masks, those annotations should be preferred over generic sky masks. In the current ADE20K mapping, `cloud` is absent, so `sky` is used as a proxy for cloud-region awareness. This can cause the detector to include clear sky areas, which the classification stage partially addresses through the `Clear sky` class.

The classification model also depends on the quality and representativeness of GCD labels. Ambiguous cloud scenes, transitional cloud forms, and multi-layer atmospheres may be difficult to assign to a single class.

## 11. Conclusion

Cloud Chaser implements a practical two-stage approach for cloud identification in general outdoor imagery. By combining ADE20K-based environmental segmentation, self-supervised cloud representation learning, and supervised GCD classification, the system balances localization robustness with meteorological specificity. The resulting pipeline is suitable for experimentation, deployment-oriented optimization, and further research into cloud-type recognition under real-world visual conditions.
