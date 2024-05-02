import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pro_path')
parser.add_argument('--hnerv_path')
args = parser.parse_args()

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def parse_data(text):
    pattern = r'\d+: psnr: (\d+\.\d+), msssim: (\d+\.\d+), lpips: (\d+\.\d+)'
    matches = re.findall(pattern, text)
    return {'psnr': [float(match[0]) for match in matches],
            'msssim': [float(match[1]) for match in matches],
            'lpips': [float(match[2]) for match in matches]}

def plot_combined_graph(data_a, data_b, title, y_label, save_path):
    plt.plot(data_a, label='proposed.txt')
    plt.plot(data_b, label='hnerv.txt')
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(save_path)
    plt.close()

# Read and parse data from the files
a_content = read_file(os.path.join(args.pro_path, 'psnr_ssim_lpips.txt'))
b_content = read_file(os.path.join(args.hnerv_path, 'psnr_ssim_lpips.txt'))

a_data = parse_data(a_content)
b_data = parse_data(b_content)

psnr_path = os.path.join(args.pro_path, 'psnr_combined_comparison.png')
ssim_path = os.path.join(args.pro_path, 'ssim_combined_comparison.png')
lpips_path = os.path.join(args.pro_path, 'lpips_combined_comparison.png')

# Plot and save combined PSNR graph
plot_combined_graph(a_data['psnr'], b_data['psnr'], 'PSNR Comparison', 'PSNR', psnr_path)

# Plot and save combined SSIM graph
plot_combined_graph(a_data['msssim'], b_data['msssim'], 'SSIM Comparison', 'SSIM', ssim_path)

# Plot and save combined LPIPS graph
plot_combined_graph(a_data['lpips'], b_data['lpips'], 'LPIPS Comparison', 'LPIPS', lpips_path)
