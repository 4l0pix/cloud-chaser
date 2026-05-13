# U-Net Restructure Implementation Plan

## Sources

1. Rabin Dhakal, "Cloud Image Segmentation using UNet Architecture in Pytorch", Medium, 2 March 2024.
2. Deyang Yin, Jinxin Wang, Kai Zhai, Jianfeng Zheng, Hao Qiang, "Ground-Based Cloud Image Segmentation Method Based on Improved U-Net", Applied Sciences, 2024.

## Goal And Implementation Status

Replace the current compact U-Net with a stronger cloud-specific segmentation architecture while keeping the production pipeline simple:

```text
RGB image -> U-Net cloud mask -> connected components -> RGB crops -> GCD classifier
```

The redesign should be tested through controlled ablations, not by replacing the model blindly.

Status: implemented as a six-model ablation suite in `local-exec/cloud_chaser/models/unet.py`, `local-exec/scripts/unet_ablation_report.py`, and `kaggle/cloud_chaser_kaggle.ipynb`. The local and Kaggle pipelines now instantiate U-Net variants through `build_unet(...)`, store architecture metadata in checkpoints, and evaluate both SWIMSEG mask quality and GCD cascade behavior.

## Baseline: Medium U-Net

The Medium implementation is a conventional supervised U-Net recipe:

```text
features = [64, 128, 256, 512]
encoder: DoubleConv + MaxPool
bottleneck: DoubleConv(512 -> 1024)
decoder: ConvTranspose2d + skip concatenation + DoubleConv
output: Conv2d(64 -> 1)
loss: BCEWithLogitsLoss
optimizer: Adam, lr=1e-4
batch_size: 16
epochs: 5 in the article example
metric: Dice score
augmentation: Resize, Rotate, HorizontalFlip, VerticalFlip, Normalize
```

### Pros

- Simple and easy to debug.
- Very close to our current implementation.
- Uses a standard PyTorch U-Net structure with skip connections.
- Good first reproducible baseline.
- Low implementation risk.

### Cons

- No cloud-specific feature enhancements.
- Transposed-convolution upsampling may introduce artifacts.
- BCE-only training can underperform on class imbalance compared with BCE + Dice.
- Limited experiment depth in the article; five epochs is a demonstration, not a production training recipe.
- Does not explicitly address thin clouds, cloud edges, local/global context, or high/low feature exchange beyond standard concatenation.

## Improved U-Net Paper

The paper proposes an encoder-decoder U-Net variant specialized for ground-based cloud segmentation.

Key modules:

```text
Encoder:
  Dilate Block x4
  each Dilate Block = dilated convolution + ASPP + dilated convolution

Skip path:
  DS Path + Im-CSAM
  DS Path = depthwise separable residual feature path
  Im-CSAM = improved channel attention + improved spatial attention

Decoder:
  bicubic interpolation upsampling
  convolution blocks after upsampling

Metrics:
  Accuracy, Precision, Recall, F1, MIoU, Error Rate
```

The paper reports improvements over traditional U-Net on SWINySEG and TCDD. On SWINySEG, the improved model reports accuracy 0.895 and MIoU 0.801 versus baseline U-Net accuracy 0.876 and MIoU 0.763. On TCDD, it reports accuracy 0.943 and MIoU 0.893 versus baseline U-Net accuracy 0.914 and MIoU 0.844.

### Pros

- Designed specifically for ground-based cloud segmentation.
- Dilated convolution increases receptive field without immediately reducing resolution.
- ASPP adds multi-scale context, useful for cloud fields that contain both large cloud masses and small cloud fragments.
- DS residual paths improve skip-connection feature exchange while keeping the path lighter than standard full convolutions.
- Attention modules focus the model on important channel/spatial cloud features.
- Bicubic upsampling may produce cleaner masks than transposed convolution.
- Paper includes ablation evidence for dilate block, DS path, and Im-CSAM.

### Cons

- More complex and higher implementation risk.
- Larger parameter size; the authors explicitly note model size as a future optimization issue.
- Paper implementation uses TensorFlow, while our stack is PyTorch, so we must reproduce modules carefully.
- More moving parts make debugging harder.
- More likely to overfit if our Kaggle subset is smaller or less diverse than SWINySEG/TCDD.
- Attention and ASPP increase memory use, which matters on Kaggle T4.

## Similarities

Both approaches:

- Use supervised semantic segmentation.
- Predict one binary cloud mask channel.
- Use an encoder-decoder topology.
- Preserve spatial detail with skip connections.
- Train from RGB cloud images and binary masks.
- Evaluate overlap with Dice/IoU-style metrics.

## Differences

| Area | Medium U-Net | Improved U-Net Paper |
|---|---|---|
| Encoder | Standard DoubleConv blocks | Dilated convolution + ASPP + dilated convolution |
| Context | Local convolutional context | Multi-scale context through ASPP |
| Skip connections | Direct concatenation | DS residual path + Im-CSAM attention before fusion |
| Upsampling | ConvTranspose2d | Bicubic interpolation plus convolution |
| Loss | BCEWithLogitsLoss in example | Paper reports metrics, does not prescribe our exact PyTorch loss |
| Complexity | Low | High |
| Expected edge quality | Moderate | Better thin-cloud and edge preservation |
| Deployment risk | Low | Medium/high |

## Recommendation

Use a staged ablation path:

1. **Baseline A: Current compact U-Net**
   - Keep current base channels 32.
   - Loss: BCEWithLogitsLoss + Dice.
   - Purpose: preserve a known working baseline.

2. **Baseline B: Medium U-Net**
   - Implement configurable `features=[64,128,256,512]`.
   - Keep BCE + Dice rather than BCE-only for production comparison.
   - Purpose: determine whether capacity alone improves SWIMSEG/GCD cascade.

3. **Model C: Dilated U-Net**
   - Replace DoubleConv encoder blocks with dilated blocks.
   - Purpose: isolate the receptive-field improvement.

4. **Model D: Dilated + ASPP U-Net**
   - Add ASPP inside encoder blocks.
   - Purpose: test multi-scale cloud context.

5. **Model E: Add bicubic decoder**
   - Replace ConvTranspose2d with bicubic interpolation + convolution.
   - Purpose: test cleaner upsampling without attention complexity.

6. **Model F: Full improved U-Net**
   - Add DS path and Im-CSAM on skip connections.
   - Purpose: reproduce the paper’s full architecture.

## Metrics To Compare

Segmentation metrics on SWIMSEG:

```text
mIoU
Dice
Precision
Recall
F1
Error Rate
inference time per image
GPU memory
parameter count
```

Pipeline metrics on GCD:

```text
segmentation gate accuracy
classifier accuracy given detection
end-to-end cascade accuracy
false clear-sky detections
missed cloudy images
```

## Implemented Steps

1. Added `unet.architecture` to `configs/default.yaml`:

```yaml
unet:
  architecture: compact
  features: [32, 64, 128, 256]
```

2. Refactored `cloud_chaser/models/unet.py`:

```text
CloudUNetCompact
CloudUNetStandard
CloudUNetDilated
CloudUNetDilatedASPP
CloudUNetImproved
build_unet(architecture, features)
```

3. Added reusable modules:

```text
DepthwiseSeparableConv
ASPP
DilateBlock
DSResidualPath
ChannelAttention
SpatialAttention
ImprovedCSAM
BicubicUpBlock
```

4. Updated training and inference to instantiate through `build_unet`.

5. Stored architecture metadata in checkpoints:

```text
architecture
features
image_size
threshold
```

6. Add an experiment script:

```bash
python scripts/unet_ablation_report.py --config configs/default.yaml
```

7. Only promote a new U-Net architecture if it improves both:

```text
SWIMSEG mIoU/Dice
GCD end-to-end cascade accuracy
```

## Is One Better?

The improved U-Net paper is likely better for final segmentation quality because it is cloud-specific and reports stronger MIoU than traditional U-Net. The Medium U-Net is better as a clean implementation baseline because it is simpler, easier to reproduce, and less likely to introduce bugs.

Practical decision:

```text
Use Medium-style U-Net as the baseline.
Adopt paper modules incrementally through ablations.
Keep only modules that improve SWIMSEG segmentation and GCD cascade accuracy enough to justify their cost.
```
