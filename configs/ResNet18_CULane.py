num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)
epochs = 16
batch_size = 24

optimizer = dict(type='AdamW', lr=0.6e-3)  # 3e-4 for batchsize 8
total_iter = (55698 // batch_size) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)
test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)
work_dirs = "work_dirs/ResNet18_culane"

net = dict(type='Detector', )
backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True)
neck = dict(type='Aggregator',
            in_channels=[512],
            out_channels=64)
heads = dict(type='CLRHead',
             num_priors=192,
             refine_layers=1,
             fc_hidden_dim=64,
             sample_points=36)

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = './data/CULane'
dataset_type = 'CULane'
diff_path = ''
threshold = 15
dataset = dict(val=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
),
test=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
))

workers = 10
log_interval = 1000
# seed = 0
num_classes = 4 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
