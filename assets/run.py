import subprocess
import os

base_path = 'data/DAVIS-data/DAVIS/JPEGImages/1080p'
name_list = ['bear', 'blackswan', 'bmx-bumps', 'bmx-trees', 
             'boat', 'breakdance', 'breakdance-flare', 'bus', 
             'camel', 'car-roundabout', 'car-shadow', 'car-turn', 'cows', 
             'dance-jump', 'dance-twirl', 'dog', 'dog-agility', 
             'drift-chicane', 'drift-straight', 'drift-turn', 
             'elephant', 'flamingo', 'goat', 'hike', 'hockey', 
             'horsejump-high', 'horsejump-low', 'kite-surf', 'kite-walk', 
             'libby', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 
             'motocross-jump', 'motorbike', 'paragliding', 'paragliding-launch', 
             'parkour', 'rhino', 'rollerblade', 'scooter-black', 'scooter-gray', 
             'soapbox', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']
model_size_list = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
epoch = 300
proposed = True
input_type = 'fft'
filter_rate = 0.8
lr = 0.002
lr_embed = '1.25_30'
msp = '20_1_5'

for name in name_list:
    outf = f'davis_{name}'
    img_path = os.path.join(base_path, name)
    for i in range(len(model_size_list)):
        model_size = model_size_list[i]
        print(name, img_path)
        '''proposed-train'''
        subprocess.run(f'python train.py \
                       --outf {outf} --exp_id {name}_{str(model_size)} --data_path {img_path} \
                       -e {epoch} --filter_rate {filter_rate} --proposed {proposed} --msp {msp} \
                       --input_type {input_type} --lr_embed {lr_embed} --modelsize {model_size} \
                       --vid {name} --conv_type convnext pshuffel --act gelu --norm none --crop_list 640_1280 \
                       --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 --dec_strds 5 4 4 2 2 \
                       --ks 0_1_5 --lr_ks 0_3_5 --reduce 1.2 --eval_freq 30 --lower_width 12 -b 1 --lr {lr}', shell=True)
        '''proposed-eval'''
        weight_path = f'output/{outf}/{name}_{str(model_size)}/model_latest.pth'
        subprocess.run(f'python train.py \
                       --outf {outf} --exp_id {name}_{str(model_size)}_eval --data_path {img_path} \
                       -e {epoch} --filter_rate {filter_rate} --proposed {proposed} --msp {msp} \
                       --input_type {input_type} --lr_embed {lr_embed} --modelsize {model_size} \
                       --vid {name} --conv_type convnext pshuffel --act gelu --norm none --crop_list 640_1280 \
                       --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 --dec_strds 5 4 4 2 2 \
                       --ks 0_1_5 --lr_ks 0_3_5 --reduce 1.2 --eval_freq 30 --lower_width 12 -b 1 --lr {lr} \
                       --eval_only --weight {weight_path} --quant_model_bit 8 --quant_embed_bit 6 --dump_images' , shell=True)
