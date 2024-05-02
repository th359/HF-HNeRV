import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--pro_path')
parser.add_argument('--hnerv_path')
args = parser.parse_args()

def open_file(text_path):
    orig_psnr = []
    orig_ssim = []
    orig_lpips = []
    quant_psnr = []
    quant_ssim = []
    quant_lpips = []
    flag = False
    with open(os.path.join(text_path, 'psnr_ssim_lpips.txt'), 'r') as file:
        for line in file:
            if 'orig' in line:
                flag = False
            elif 'quant' in line:
                flag = True
            else:
                psnr = float(re.search(r'psnr:\s*([\d.]+)', line).group(1))
                ssim = float(re.search(r'msssim:\s*([\d.]+)', line).group(1))
                lipis = float(re.search(r'lpips:\s*([\d.]+)', line).group(1))
                if flag:
                    #quant
                    quant_psnr.append(psnr)
                    quant_ssim.append(ssim)
                    quant_lpips.append(lipis)
                else:
                    #orig
                    orig_psnr.append(psnr)
                    orig_ssim.append(ssim)
                    orig_lpips.append(lipis)
    return orig_psnr, orig_ssim, orig_lpips, quant_psnr, quant_ssim, quant_lpips

def cal_mean(l):
    mean = round(statistics.mean(l), 4)
    return mean

pop, pos, pol, qop, qos, qol = open_file(args.pro_path)
php, phs, phl, qhp, qhs, qhl = open_file(args.hnerv_path)

mpop = cal_mean(pop)
mpos = cal_mean(pos)
mpol = cal_mean(pol)
mqop = cal_mean(qop)
mqos = cal_mean(qos)
mqol = cal_mean(qol)

mphp = cal_mean(php)
mphs = cal_mean(phs)
mphl = cal_mean(phl)
mqhp = cal_mean(qhp)
mqhs = cal_mean(qhs)
mqhl = cal_mean(qhl)

flag_propo = 'Propo'
flag_hnerv = 'HNeRV'
with open(os.path.join(args.pro_path, 'compare.txt'), 'w') as f:
    f.write('psnr\n')
    f.write(f'propo_orig: {str(mpop)}\nhnerv_orig: {str(mphp)}\n')
    f.write(f'propo_quan: {str(mqop)}\nhnerv_quan: {str(mqhp)}\n')
    f.write('msssim\n')
    f.write(f'propo_orig: {str(mpos)}\nhnerv_orig: {str(mphs)}\n')
    f.write(f'propo_quan: {str(mqos)}\nhnerv_quan: {str(mqhs)}\n')
    f.write('lpips\n')
    f.write(f'propo_orig: {str(mpol)}\nhnerv_orig: {str(mphl)}\n')
    f.write(f'propo_quan: {str(mqol)}\nhnerv_quan: {str(mqhl)}\n')
    f.write('\norig\n')
    f.write(f'psnr  : {flag_propo if mpop >= mphp else flag_hnerv}\n')
    f.write(f'msssim: {flag_propo if mpos >= mphs else flag_hnerv}\n')
    f.write(f'lipis : {flag_propo if mpol <= mphl else flag_hnerv}\n')
    f.write('\nquant\n')
    f.write(f'psnr  : {flag_propo if mqop >= mqhp else flag_hnerv}\n')
    f.write(f'msssim: {flag_propo if mqos >= mqhs else flag_hnerv}\n')
    f.write(f'lipis : {flag_propo if mqol <= mqhl else flag_hnerv}\n')