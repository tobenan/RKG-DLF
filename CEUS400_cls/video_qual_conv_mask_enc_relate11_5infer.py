num_segs = 32
# scale_rate = 10

model = dict(
    type='RecoginizerTwoStream_DAFT_QUAL_5infer',
    mlc_loss = 'ml_softmax', #  ml_softmax, asl, calibrated_hinge,softmargin,calibrated_hinge
    backbone=dict(
        type='ResNet',
        pretrained=None,
        depth=18,
        out_indices=(0,1,2,3),
        norm_eval=False),
    fuse_cfg = dict(
        type='FuseNet',
        depth=18),
    cls_head_CEUS=dict(
        type='TSN_FuseHead_conv_mask_fusion_qual_relate11',
        num_classes=3+3, 
        in_channels=512,
        fuse_channels=14, 
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),#AvgConsensus Att
        dropout_ratio=0.5, 
        init_std=0.001),
    cls_head_B=dict(
        type='TSN_FuseHead_conv_mask_fusion_qual_relate11',
        num_classes=3+2, 
        in_channels=512,
        fuse_channels=40,  
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))

dataset_type = 'CEUSDatsaset'
data_root = '.'
data_root_val = '.'
ann_file_train = f'data/preprocessing3/preprocess_list_cleanlab_train182.txt'
ann_file_val = f'data/preprocessing3/preprocess_list_cleanlab_test182.txt'
ann_file_test = f'data/preprocessing3/preprocess_list_cleanlab_test182.txt'
coarse_seg = 'data/preprocessing3/coarse_seg'

norm_json = 'data/preprocessing3/video_norms_cut_2mode.json'
ceus_quan = '/media/ders/GDH/mmaction-master/demo/tabular/ceus_quan_7ori.xlsx'
bus_quan = '/media/ders/GDH/mmaction-master/demo/tabular/bus_quan_ori.xlsx'
qual ="/media/ders/GDH/mmaction-master/data/qual_more.xlsx"
# tumor_bbox = None

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=32),
    dict(type='RawFrameDecode'),
    dict(type='GetCoarseSeg'),
    dict(type='ConcatModeCut'),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Flip', flip_ratio=0.5, direction='vertical',),
    dict(type='Imgaug', transforms=[
            dict(type='Rotate', rotate=(-20, 20))
        ]),
    dict(type='VideoNorm'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='DiffImgs'),
    dict(type='SplitCEUS'),
    dict(type='FormatShapeCEUS', input_format='NCHW'),
    dict(type='Collect', keys=['img_ceus', 'img_b', 'coarse_seg','label',"ceus_quan", "bus_quan","qual"], meta_keys=[]),
    dict(type='ToTensor', keys=['img_ceus', 'img_b', 'coarse_seg','label',"ceus_quan", "bus_quan","qual"]),
    
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=32,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='GetCoarseSeg'),
    dict(type='ConcatModeCut'),
    dict(type='VideoNorm'),
    dict(type='SplitCEUS'),
    dict(type='FormatShapeCEUS', input_format='NCHW'),
    dict(type='Collect', keys=['img_ceus', 'img_b', 'coarse_seg','label',"ceus_quan", "bus_quan","qual"], meta_keys=[]),
    dict(type='ToTensor', keys=['img_ceus', 'img_b', 'coarse_seg','label',"ceus_quan", "bus_quan","qual"]),

]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    train_dataloader=dict(drop_last=True),
    test_dataloader=dict(videos_per_gpu=8),
    train=dict(
        type='ClassBalancedDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            data_prefix=data_root,
            coarse_seg=coarse_seg,
            norm_json = norm_json,
            bus_quan =bus_quan,
            ceus_quan =ceus_quan,
            qual = qual,
            tabquannorm = False, ####
            tabquan_enc = 'Q_PLE',#### Q_PLE 
            n_bins=2,
            pipeline=train_pipeline),
        oversample_thr=1
    ),

    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        norm_json = norm_json,
        coarse_seg=coarse_seg,
        bus_quan =bus_quan,
        ceus_quan =ceus_quan,
        qual = qual,
        tabquannorm = False, ####
        tabquan_enc = 'Q_PLE',#### Q_PLE
        n_bins=2,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        norm_json = norm_json,
        coarse_seg=coarse_seg,
        bus_quan =bus_quan,
        ceus_quan =ceus_quan,
        qual = qual,
        tabquannorm = False, ####
        tabquan_enc = 'Q_PLE',####
        n_bins=2,
        pipeline=test_pipeline))


evaluation = dict(
    interval=1,  metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix','ceus_metric'])
# save_best='top_1_acc',
optimizer = dict(
    type='SGD', lr=1e-3/2, momentum=0.9,
    weight_decay=0.001)
# optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3, amsgrad=False)
# 1e-3/2
# random seed
# seed = 1797512715

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True)
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=1e-5)
total_epochs = 200 ##############
log_config = dict(interval=1,hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# find_unused_parameters = True
checkpoint_config = dict(interval=100)
gpu_ids=[0]


# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 4
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
