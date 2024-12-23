# RKG-DLF

## Install
Please follow the instruction in docs to install program. Thanks for the contribution of mmaction.

## Data
train\val\test.txt should be organized as

" \\..\\..\\.jpg(path_sample1) 32(frame_number) label "

## Run
CUDA_VISIBLE_DEVICES="3,4" PORT=23400 tools/dist_train.sh CEUS400_cls/video_qual_conv_mask_enc_relate11_5infer.py 2 --validate
