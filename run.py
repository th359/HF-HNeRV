import subprocess
import os

outf = 'davis'
epoch = 300
fix_epoch = 150
model_size = 1.5
davis_path = 'data/DAVIS-data/DAVIS/JPEGImages/1080p'
davis_folders = [f.path for f in os.scandir(davis_path) if f.is_dir()]
davis_folders.sort()

for i in range(len(davis_folders)):
    img_path = davis_folders[i]
    name = os.path.basename(img_path)
    print(name, img_path)
    #HF-NeRV
    subprocess.run(f'python train_nerv_all.py -e {epoch} --fix_epoch {fix_epoch} --data_path {img_path} --exp_id {name} --outf {outf} --modelsize {model_size} --use_hnerv --propose --use_highpass', shell=True)
    eval_path = f'output/{outf}/{name}/model_latest.pth'
    subprocess.run(f'python train_nerv_all.py -e {epoch} --fix_epoch {fix_epoch} --data_path {img_path} --exp_id {name}_eval --outf {outf} --modelsize {model_size} --propose --use_highpass --eval_only --weight {eval_path} --dump_images --dump_videos', shell=True)
    #HNeRV
    subprocess.run(f'python train_nerv_all.py -e {epoch} --fix_epoch {fix_epoch} --data_path {img_path} --exp_id {name}_hnerv --outf {outf} --modelsize {model_size} --use_hnerv', shell=True)
    hnerv_eval_path = f'output/{outf}/{name}_hnerv/model_latest.pth'
    subprocess.run(f'python train_nerv_all.py -e {epoch} --fix_epoch {fix_epoch} --data_path {img_path} --exp_id {name}_hnerv_eval --outf {outf} --modelsize {model_size} --use_hnerv --eval_only --weight {hnerv_eval_path} --dump_images --dump_videos', shell=True)
    #val.py
    pro_path = f'output/{outf}/{name}_eval'
    hnerv_path = f'output/{outf}/{name}_hnerv_eval'
    subprocess.run(f'python val.py --pro_path {pro_path} --hnerv_path {hnerv_path}', shell=True)
    subprocess.run(f'python compare.py --pro_path {pro_path} --hnerv_path {hnerv_path}', shell=True)
subprocess.run(f'python excel.py --path output/{outf}', shell=True)