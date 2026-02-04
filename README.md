#  Context-aware Affordance Grounding

This is the PyTorch implementation of "Affordance Grounding in Contextual Interaction Relations". This code was written based on [https://github.com/woyut/WSAG-PLSP.git](https://github.com/woyut/WSAG-PLSP.git).

## Data preparation

1. Please follow [LOCATE](https://github.com/Reagan1311/LOCATE) to prepare the AGD20K datasets. 

2. Please download the pre-trained ViT-B/16 weights from [CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt), which is used to initialize the visual encoder during training.

3. Please donwload the checkpoint for this study from [here](). 

## Requirements

- torch=1.10.0+cu111 
- torchvision=0.11.0+cu111
- numpy
- opencv-python
- PyYAML
- tqdm
- Pillow

## Training

To train the model, run:

```
python train_context_aware.py --config context_aware_seen.yaml --seed 10000
```

Please modify the paths in the .yaml files to the locations of the datasets, pre-trained ViT weights, and `work_dir`.

Our trained model can be found
- [wsag-plsp](https://drive.google.com/drive/folders/1JaX-4w9mH0IrxtKCowoBwCbD6loMyzKe?usp=sharing).
- [ours](https://drive.google.com/drive/folders/1HxDuTtV1ZaZSLPV8bjwIrCnRDsnIa0lv?usp=sharing)

## Testing

To evaluate the trained model, run:

```
python eval_context_aware.py --config configs/context_aware_seen_test.yaml
```

Please modify the paths in the .yaml files to the locations of the datasets, pre-trained ViT weights, and `work_dir`.

And change `split_type` "Unseen" if you want to inference unseen action.


## Acknowledgement

We sincerely thank the codebase of [CLIP](https://github.com/openai/CLIP/), [SAM](https://github.com/facebookresearch/segment-anything/tree/main), [LOCATE](https://github.com/Reagan1311/LOCATE),[grounded-segment-any-parts](https://github.com/Saiyan-World/grounded-segment-any-parts), and [WSAG-PLSP](https://github.com/woyut/WSAG-PLSP).
