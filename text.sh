python tools/build_lasher_text_cache.py \
  --seq-list \
  /data/pudata/LasHeR/TrainingSet/train/trainingsetList.txt \
  --output /data/wsk/project/pixeltext/cache/lashertrain_text_features.pt \
  --provider open_clip \
  --open-clip-model ViT-B-32 \
  --open-clip-pretrained /data/wsk/project/pixeltext/pretrained/open_clip_pytorch_model.bin
