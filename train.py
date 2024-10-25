import imageio
# from pygifsicle import optimize
import argparse
import os
import random
import shutil
from datetime import datetime
import numpy as np
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from model_all import VideoDataSet, HNeRV, HNeRVDecoder, TransformInput
from model_all import VDS_Proposed
from hnerv_utils import *
from torch.utils.data import Subset
from copy import deepcopy
from dahuffman import HuffmanCodec
from torchvision.utils import save_image
import pandas as pd
from PIL import Image
import json
from math import prod
from torchinfo import summary

def main():
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='', help='data path for vid')
    parser.add_argument('--vid', type=str, default='k400_train0', help='video id',)
    parser.add_argument('--shuffle_data', action='store_true', help='randomly shuffle the frame idx')
    parser.add_argument('--data_split', type=str, default='1_1_1', 
        help='Valid_train/total_train/all data split, e.g., 18_19_20 means for every 20 samples, the first 19 samples is full train set, and the first 18 samples is chose currently')
    parser.add_argument('--crop_list', type=str, default='640_1280', help='video crop size',)
    parser.add_argument('--resize_list', type=str, default='-1', help='video resize size',)

    # NERV architecture parameters
    # Embedding and encoding parameters
    parser.add_argument('--embed', type=str, default='', help='empty string for HNeRV, and base value/embed_length for NeRV position encoding')
    parser.add_argument('--ks', type=str, default='0_3_3', help='kernel size for encoder and decoder')
    parser.add_argument('--enc_strds', type=int, nargs='+', default=[], help='stride list for encoder')
    parser.add_argument('--enc_dim', type=str, default='64_16', help='enc latent dim and embedding ratio')
    parser.add_argument('--modelsize', type=float,  default=1.5, help='model parameters size: model size + embedding parameters')
    parser.add_argument('--saturate_stages', type=int, default=-1, help='saturate stages for model size computation')

    # Decoding parameters: FC + Conv
    parser.add_argument('--fc_hw', type=str, default='9_16', help='out size (h,w) for mlp')
    parser.add_argument('--reduce', type=float, default=1.2, help='chanel reduction for next stage')
    parser.add_argument('--lower_width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list for decoder')
    parser.add_argument('--num_blks', type=str, default='1_1', help='block number for encoder and decoder')
    parser.add_argument("--conv_type", default=['convnext', 'pshuffel'], type=str, nargs="+",
        help='conv type for encoder/decoder', choices=['pshuffel', 'conv', 'convnext', 'interpolate'])
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', 
        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--start_epoch', type=int, default=-1, help='starting epoch')
    parser.add_argument('--not_resume', action='store_true', help='not resume from latest checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Epoch number')
    parser.add_argument('--block_params', type=str, default='1_1', help='residual blocks and percentile to save')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate type, default=cosine')
    parser.add_argument('--loss', type=str, default='Fusion6', help='loss type, default=L2')
    parser.add_argument('--out_bias', default='tanh', type=str, help='using sigmoid/tanh/0.5 for output prediction')

    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=10, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_model_bit', type=int, default=8, help='bit length for model quantization')
    parser.add_argument('--quant_embed_bit', type=int, default=6, help='bit length for embedding quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--dump_videos', action='store_true', default=False, help='concat the prediction images into video')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')
    parser.add_argument('--encoder_file',  default='', type=str, help='specify the embedding file')

    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, 
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")

    parser.add_argument('--exp_id', type=str, default='exp_id')
    parser.add_argument('--filter_rate', default=0.8, type=float)
    parser.add_argument('--size_n', default=128)
    parser.add_argument('--proposed', default=True, type=bool)
    parser.add_argument('--input_type', default='fft', type=str, choices=['img', 'fft', 'residue'])
    parser.add_argument('--lr_embed', type=str, default='1.25_30')
    parser.add_argument('--msp', default='20_1_5', type=str, help='model size percentage, hr_lr')
    parser.add_argument('--lr_dec_strds', type=int, nargs='+', default=[4, 4, 4, 2], help='strides list for lr_decoder')
    parser.add_argument('--lr_ks', type=str, default='0_3_5', help='kernel size for LR_Decoder1/2')

    args = parser.parse_args()
    torch.set_printoptions(precision=4) 
    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    args.enc_strd_str, args.dec_strd_str = ','.join([str(x) for x in args.enc_strds]), ','.join([str(x) for x in args.dec_strds])
    args.quant_str = f'quant_M{args.quant_model_bit}_E{args.quant_embed_bit}'

    args.outf = os.path.join(args.outf, args.exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)
    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    train(None, args)

def data_to_gpu(x, device):
    return x.to(device)

def train(local_rank, args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    args.metric_names = ['pred_seen_psnr', 'pred_seen_ssim', 'pred_unseen_psnr', 'pred_unseen_ssim',
        'quant_seen_psnr', 'quant_seen_ssim', 'quant_unseen_psnr', 'quant_unseen_ssim']
    best_metric_list = [torch.tensor(0) for _ in range(len(args.metric_names))]

    # setup dataloader    
    if args.proposed:
        full_dataset = VDS_Proposed(args)
    else:
        full_dataset = VideoDataSet(args)

    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batchSize, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False, worker_init_fn=worker_init_fn)
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)
    args.dump_vis = (args.dump_images or args.dump_videos)

    #  Make sure the testing dataset is fixed for every run
    train_dataset =  Subset(full_dataset, train_ind_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True,
         num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True, worker_init_fn=worker_init_fn)

    # Compute the parameter number
    total_enc_strds = np.prod(args.enc_strds)
    embed_hw = args.final_size / total_enc_strds**2
    enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split('_')]
    embed_dim = int(embed_ratio * args.modelsize * 1e6 / args.full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio) 
    embed_param = float(embed_dim) / total_enc_strds**2 * args.final_size * args.full_data_length
    args.enc_dim = f'{int(enc_dim1)}_{embed_dim}' 
    fc_param = (np.prod(args.enc_strds) // np.prod(args.dec_strds))**2 * 9

    p_hr, p_lr1, p_lr2 = [float(x) for x in args.msp.split('_')]
    p_sum = float(p_hr + p_lr1 + p_lr2)
    decoder_size = args.modelsize * 1e6 - embed_param
    decoder_size_hr = decoder_size * p_hr / p_sum
    decoder_size_lr1 = decoder_size * p_lr1 / p_sum
    decoder_size_lr2 = decoder_size * p_lr2 / p_sum

    '''calculate HR_decoder fc_dim by decoder_size_hr'''
    ch_reduce = 1. / args.reduce
    dec_ks1, dec_ks2 = [int(x) for x in args.ks.split('_')[1:]]
    fix_ch_stages = len(args.dec_strds) if args.saturate_stages == -1 else args.saturate_stages
    a =  ch_reduce * sum([ch_reduce**(2*i) * s**2 * min((2*i + dec_ks1), dec_ks2)**2 for i,s in enumerate(args.dec_strds[:fix_ch_stages])])
    b =  embed_dim * fc_param 
    c =  args.lower_width **2 * sum([s**2 * min(2*(fix_ch_stages + i) + dec_ks1, dec_ks2)  **2 for i, s in enumerate(args.dec_strds[fix_ch_stages:])])
    args.fc_dim = int(np.roots([a,b,c - decoder_size_hr]).max())
    '''calculate LR_decoder fc_dim by decoder_size_hr'''
    _, pe_len = [float(x) for x in args.lr_embed.split('_')] 
    args.fc_pe_1 = int(pe_len*2 / 3) 
    args.fc_pe_2 = int(pe_len*2 - args.fc_pe_1) 
    args.fc_lr = args.final_size / prod([i**2 for i in args.lr_dec_strds]) 
    lr_ch_stages = len(args.lr_dec_strds) 
    lrdec_ks1, lrdec_ks2 = [int(x) for x in args.lr_ks.split('_')[1:]]
    a =  ch_reduce * sum([ch_reduce**(2*i) * s**2 * min((2*i + lrdec_ks1), lrdec_ks2)**2 for i,s in enumerate(args.lr_dec_strds[:lr_ch_stages])])
    b = (args.fc_pe_1*args.fc_lr)+args.fc_lr
    args.lr_fc_dim1 = int(np.roots([a, b, -decoder_size_lr1]).max())
    b = (args.fc_pe_2*args.fc_lr)+args.fc_lr
    args.lr_fc_dim2 = int(np.roots([a, b, -decoder_size_lr2]).max())

    if args.proposed:
        PE = PositionalEncoding(args.lr_embed)
        args.embed_length = PE.embed_length

    # Building model
    model = HNeRV(args)

    ##### get model params and flops #####
    model_param = (sum([p.data.nelement() for p in model.parameters()]) / 1e6)
    encoder_param = (sum([p.data.nelement() for p in model.encoder.parameters()]) / 1e6) 
    decoder_param = (sum([p.data.nelement() for p in model.decoder.parameters()]) / 1e6) 
    lr_decoder1_param = (sum([p.data.nelement() for p in model.lr_decoder1.parameters()]) / 1e6) 
    lr_decoder2_param = (sum([p.data.nelement() for p in model.lr_decoder2.parameters()]) / 1e6) 
    total_param = decoder_param + lr_decoder1_param + lr_decoder2_param + embed_param / 1e6
    args.encoder_param, args.decoder_param, args.total_param = encoder_param, decoder_param, total_param
    mdl_str = f'ModelPr_{round(model_param, 4)}M'
    enc_str = f'Encoder_{round(encoder_param, 4)}M'
    dec_str = f'Decoder_{round(decoder_param, 4)}M'
    dec1_str= f'LR_Dec1_{round(lr_decoder1_param, 4)}M'
    dec2_str= f'LR_Dec2_{round(lr_decoder2_param, 4)}M'
    lrt_str = f'LR_Toal_{round(lr_decoder1_param+lr_decoder2_param, 4)}M'
    tal_str = f'TotalPr_{round(total_param, 4)}M'
    param_str = f'{mdl_str}_{enc_str}_{dec_str}_{dec1_str}_{dec2_str}_{lrt_str}_{tal_str}'
    
    print(f'enc_param {round(encoder_param, 4)}M')
    print(f'dec_param {round(decoder_param, 4)}M')
    print(f'lrt_param {round(lr_decoder1_param+lr_decoder2_param, 4)}M')
    print(f'lr1_param {round(lr_decoder1_param, 4)}M')
    print(f'lr2_param {round(lr_decoder2_param, 4)}M')
    print(f'tal_param {round(total_param, 4)}M')

    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
        f.write(str(model) + '\n' + f'{mdl_str}\n'+ f'{enc_str}\n'+ f'{dec_str}\n'+ f'{dec1_str}\n'+ f'{dec2_str}\n'+ f'{lrt_str}\n'+ f'{tal_str}\n')
    with open('{}/params.json'.format(args.outf), 'a') as f:
        json.dump(args.__dict__, f, indent=4)
    writer = SummaryWriter(os.path.join(args.outf, param_str, 'tensorboard'))

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), weight_decay=0.)
    args.transform_func = TransformInput(args)

    # resume from args.weight
    checkpoint = None
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
            model.load_state_dict(new_ckt, strict=False)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt, strict=False)
        else:
            model.load_state_dict(new_ckt, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))        

    # resume from model_latest
    if not args.not_resume:
        checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    if args.start_epoch < 0:
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] 
        args.start_epoch = max(args.start_epoch, 0)

    if args.eval_only:
        print_str = 'Evaluation ... \n {} Results for checkpoint: {}\n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.weight)
        results_list, hw = evaluate(model, full_dataloader, local_rank, args, args.dump_vis, huffman_coding=True, PE=PE)
        print_str = f'PSNR for output {hw} for quant {args.quant_str}: '
        for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
            best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
            cur_v = RoundTensor(best_metric_value, 2 if 'psnr' in metric_name else 4)
            print_str += f'best_{metric_name}: {cur_v} | '
            best_metric_list[i] = best_metric_value
        if local_rank in [0, None]:
            print(print_str, flush=True)
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n\n')        
            args.train_time, args.cur_epoch = 0, args.epochs
            Dump2CSV(args, best_metric_list, results_list, [torch.tensor(0)], 'eval.csv')

        return

    # Training
    start = datetime.now()

    psnr_list = []
    for epoch in range(args.start_epoch, args.epochs):
        model.train()       
        epoch_start_time = datetime.now()
        pred_psnr_list = []
        # iterate over dataloader
        device = next(model.parameters()).device
        for i, sample in enumerate(train_dataloader):
            if args.proposed:
                # ----- Proposed Method (use VDS_Proposed) -----
                img_data = data_to_gpu(sample['img'], device)
                img_idx = data_to_gpu(sample['idx'], device)
                norm_idx = data_to_gpu(sample['norm_idx'], device)
                fft_data = data_to_gpu(sample['fft'], device)
                lr_data = data_to_gpu(sample['lr'], device)
                resize_data = data_to_gpu(sample['resize'], device)

                # forward and backward
                img_gt = img_data
                if args.input_type == 'img':
                    cur_input = img_data
                elif args.input_type == 'fft':
                    cur_input = fft_data
                elif args.input_type == 'residue':
                    cur_input = img_gt - lr_data

                embed_input = PE(norm_idx).float()

                cur_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
                lr = adjust_lr(optimizer, cur_epoch, args)
                img_out, residue, out1, lr_out1, lr_out2, _, _ = model(cur_input, embed_input=embed_input)
                final_loss = loss_fn(img_out, img_gt, args.loss)
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()

                if 5 == img_idx and (epoch % 15 == 0 or epoch == args.epochs-1) :
                    residue_gt = img_gt - lr_data
                    diff = residue_gt - residue
                    l = [(img_gt, 0), (img_out, 0), (residue, 1), (out1, 0), (lr_out2, 1), (lr_out1, 0)]
                    for idx in range(len(l)):
                        d = l[idx][0].squeeze(0)
                        d = d.to('cpu').detach().numpy().copy()
                        if l[idx][1] == 0:
                            d = np.clip(d, 0, 1)
                        if l[idx][1] == 1:
                            d = np.abs(d)
                        d = (d*255).astype(np.uint8)
                        d = d.transpose(1, 2, 0)
                        d = Image.fromarray(d)
                        if idx == 0:
                            dst = Image.new('RGB', (d.width*len(l), d.height))
                        dst.paste(d, (d.width*idx, 0))
                    log_path = os.path.join(args.outf, 'log')
                    if not os.path.exists(log_path):
                        os.makedirs(log_path)
                    dst.save(f'{log_path}/{epoch}_{i}.png')

                pred_psnr_list.append(psnr_fn_single(img_out.detach(), img_gt)) 
                if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                    pred_psnr = torch.cat(pred_psnr_list).mean()
                    print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} pred_PSNR: {}'.format(
                        datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                        RoundTensor(pred_psnr, 2))
                    print(print_str, flush=True)
                    if local_rank in [0, None]:
                        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                            f.write(print_str + '\n')
            else:
                # ----- HNeRV (use VideoDataSet) -----
                img_data = data_to_gpu(sample['img'], device)
                norm_idx = data_to_gpu(sample['norm_idx'], device)
                img_idx  = data_to_gpu(sample['idx'], device)

                # forward and backward
                img_data, img_gt, inpaint_mask = args.transform_func(img_data)
                cur_input = img_data
                cur_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
                lr = adjust_lr(optimizer, cur_epoch, args)
                img_out, _,  _, _ = model(cur_input)
                final_loss = loss_fn(img_out, img_gt, args.loss)      
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()

                pred_psnr_list.append(psnr_fn_single(img_out.detach(), img_gt)) 
                if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                    pred_psnr = torch.cat(pred_psnr_list).mean()
                    print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} pred_PSNR: {}'.format(
                        datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                        RoundTensor(pred_psnr, 2))
                    print(print_str, flush=True)
                    if local_rank in [0, None]:
                        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                            f.write(print_str + '\n')

        # ADD train_PSNR TO TENSORBOARD
        if local_rank in [0, None]:
            h, w = img_out.shape[-2:]
            writer.add_scalar(f'Train/pred_PSNR_{h}X{w}', pred_psnr, epoch+1)
            writer.add_scalar('Train/lr', lr, epoch+1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1, 3, 5]:
            results_list, hw = evaluate(model, full_dataloader, local_rank, args, 
                args.dump_vis if epoch == args.epochs - 1 else False, 
                True if epoch == args.epochs - 1 else False, PE=PE)            
            if local_rank in [0, None]:
                # ADD val_PSNR TO TENSORBOARD
                print_str = f'Eval at epoch {epoch+1} for {hw}: '
                for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
                    best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
                    if 'psnr' in metric_name:
                        writer.add_scalar(f'Val/{metric_name}_{hw}', metric_value.max(), epoch+1)
                        writer.add_scalar(f'Val/best_{metric_name}_{hw}', best_metric_value, epoch+1)
                        if metric_name == 'pred_seen_psnr':
                            psnr_list.append(metric_value.max())
                        print_str += f'{metric_name}: {RoundTensor(metric_value, 2)} | '
                    best_metric_list[i] = best_metric_value
                print(print_str, flush=True)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),   
        }    
        if local_rank in [0, None]:
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if (epoch + 1) % args.epochs == 0:
                print('------final epoch-------')
                args.cur_epoch = epoch + 1
                args.train_time = str(datetime.now() - start)
                Dump2CSV(args, best_metric_list, results_list, psnr_list, f'epoch{epoch+1}.csv')
                torch.save(save_checkpoint, f'{args.outf}/epoch{epoch+1}.pth')
                if best_metric_list[0]==results_list[0]:
                    torch.save(save_checkpoint, f'{args.outf}/model_best.pth')
                    print(f'true: {best_metric_list[0]} == {results_list[0]}')
                else:
                    print(f'false: {best_metric_list[0]} != {results_list[0]}')

    if local_rank in [0, None]:
        print(f"Training complete in: {str(datetime.now() - start)}")


# Writing final results in CSV file
def Dump2CSV(args, best_results_list, results_list, psnr_list, filename='results.csv'):
    result_dict = {'Vid':args.vid, 'CurEpoch':args.cur_epoch, 'Time':args.train_time, 
        'FPS':args.fps, 'Split':args.data_split, 'Embed':args.embed, 'Crop': args.crop_list,
        'Resize':args.resize_list, 'Lr_type':args.lr_type, 'LR (E-3)': args.lr*1e3, 'Batch':args.batchSize,
        'Size (M)': f'{round(args.encoder_param, 2)}_{round(args.decoder_param, 2)}_{round(args.total_param, 2)}', 
        'ModelSize': args.modelsize, 'Epoch':args.epochs, 'Loss':args.loss, 'Act':args.act, 'Norm':args.norm,
        'FC':args.fc_hw, 'Reduce':args.reduce, 'ENC_type':args.conv_type[0], 'ENC_strds':args.enc_strd_str, 'KS':args.ks,
        'enc_dim':args.enc_dim, 'DEC':args.conv_type[1], 'DEC_strds':args.dec_strd_str, 'lower_width':args.lower_width,
         'Quant':args.quant_str, 'bits/param':args.bits_per_param, 'bits/param w/ overhead':args.full_bits_per_param, 
        'bits/pixel':args.total_bpp, f'PSNR_list_{args.eval_freq}':','.join([RoundTensor(v, 2) for v in psnr_list]),}
    result_dict.update({f'best_{k}':RoundTensor(v, 4 if 'ssim' in k else 2) for k,v in zip(args.metric_names, best_results_list)})
    result_dict.update({f'{k}':RoundTensor(v, 4 if 'ssim' in k else 2) for k,v in zip(args.metric_names, results_list) if 'pred' in k})
    csv_path = os.path.join(args.outf, filename)
    print(f'results dumped to {csv_path}')
    pd.DataFrame(result_dict,index=[0]).to_csv(csv_path)


@torch.no_grad()
def evaluate(model, full_dataloader, local_rank, args, 
    dump_vis=False, huffman_coding=False, PE=None):
    img_embed_list = []
    model_list, quant_ckt = quant_model(model, args)
    metric_list = [[] for _ in range(len(args.metric_names))]
    for model_ind, cur_model in enumerate(model_list):
        time_list = []
        cur_model.eval()
        device = next(cur_model.parameters()).device
        if dump_vis:
            visual_dir = f'{args.outf}/visualize_model' + ('_quant' if model_ind else '_orig')
            print(f'Saving predictions to {visual_dir}...')
            if not os.path.isdir(visual_dir):
                os.makedirs(visual_dir)       
            visual_txt = 'quant' if model_ind else 'orig'
            with open('{}/psnr_ssim_lpips.txt'.format(args.outf), 'a') as f:
                f.write(visual_txt + '\n') 

        for i, sample in enumerate(full_dataloader):
            if args.proposed:
                # ----- Proposed Method (use VDS_Proposed) -----
                img_data = data_to_gpu(sample['img'], device)
                img_idx = data_to_gpu(sample['idx'], device)
                norm_idx = data_to_gpu(sample['norm_idx'], device)
                fft_data = data_to_gpu(sample['fft'], device)
                lr_data = data_to_gpu(sample['lr'], device)
                resize_data = data_to_gpu(sample['resize'], device)

                img_gt = img_data
                if args.input_type == 'img':
                    cur_input = img_data
                elif args.input_type == 'fft':
                    cur_input = fft_data
                elif args.input_type == 'fft':
                    cur_input = img_gt - lr_data
                embed_input = PE(norm_idx).float()
                #img_out, residue, embed_list, dec_time = cur_model(cur_input, dequant_vid_embed[i] if model_ind else None, lr_data=lr_data)
                img_out, residue, out1, lr_out1, lr_out2, embed_list, dec_time = cur_model(cur_input, dequant_vid_embed[i] if model_ind else None, embed_input=embed_input)

            else:
                # ----- HNeRV (use VideoDataSet) -----
                img_data = data_to_gpu(sample['img'], device)
                norm_idx = data_to_gpu(sample['norm_idx'], device)
                img_idx  = data_to_gpu(sample['idx'], device)

                img_data, img_gt, inpaint_mask = args.transform_func(img_data)
                cur_input = norm_idx if 'pe' in args.embed else img_data
                img_out, _, embed_list, dec_time = cur_model(cur_input, dequant_vid_embed[i] if model_ind else None)

            if model_ind == 0:
                img_embed_list.append(embed_list[0])
            
            # collect decoding fps
            time_list.append(dec_time)
            if args.eval_fps:
                time_list.pop()
                for _ in range(100):
                    img_out, embed_list, dec_time = cur_model(cur_input, embed_list[0])
                    time_list.append(dec_time)

            # compute psnr and ms-ssim
            pred_psnr, pred_ssim = psnr_fn_batch([img_out], img_gt), msssim_fn_batch([img_out], img_gt)
            for metric_idx, cur_v in  enumerate([pred_psnr, pred_ssim]):
                for batch_i, cur_img_idx in enumerate(img_idx):
                    metric_idx_start = 2 if cur_img_idx in args.val_ind_list else 0
                    metric_list[metric_idx_start+metric_idx+4*model_ind].append(cur_v[:,batch_i])

            # dump predictions
            if dump_vis:
                for batch_ind, cur_img_idx in enumerate(img_idx):
                    full_ind = i * args.batchSize + batch_ind
                    dump_img_list = [img_data[batch_ind], img_out[batch_ind]]
                    # dump_img_list = [img_data[batch_ind], img_out[batch_ind], residue.squeeze(0), out1.squeeze(0)]
                    temp_psnr_list = ','.join([str(round(x[batch_ind].item(), 2)) for x in pred_psnr])
                    temp_msssim_list = ','.join([str(round(x[batch_ind].item(), 4)) for x in pred_ssim])
                    pred_lpips = lpips_fn_single(img_out[batch_ind], img_data[batch_ind])
                    temp_lpips = str(round(pred_lpips, 4))
                    with open('{}/psnr_ssim_lpips.txt'.format(args.outf), 'a') as f:
                        f.write(str(img_idx.item()).zfill(3) + ': psnr: ' + temp_psnr_list + ', msssim: ' + temp_msssim_list + ', lpips: ' + temp_lpips + '\n')
                    concat_img = torch.cat(dump_img_list, dim=2)    #img_out[batch_ind], 
                    if model_ind == False:
                        save_image(concat_img, f'{visual_dir}/pred_{full_ind:04d}_{temp_psnr_list}.png')

            # print eval results and add to log txt
            if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
                avg_time = sum(time_list) / len(time_list)
                fps = args.batchSize / avg_time
                print_str = '[{}] Rank:{}, Eval at Step [{}/{}] , FPS {}, '.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, i+1, len(full_dataloader), round(fps, 1))
                metric_name = ('quant' if model_ind else 'pred') + '_seen_psnr'
                for v_name, v_list in zip(args.metric_names, metric_list):
                    if metric_name in v_name:
                        cur_value = torch.stack(v_list, dim=-1).mean(-1) if len(v_list) else torch.zeros(1)
                        print_str += f'{v_name}: {RoundTensor(cur_value, 2)} | '
                if local_rank in [0, None]:
                    print(print_str, flush=True)
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')
        
        # embedding quantization
        if model_ind == 0:
            vid_embed = torch.cat(img_embed_list, 0) 
            quant_embed, dequant_emved = quant_tensor(vid_embed, args.quant_embed_bit)
            dequant_vid_embed = dequant_emved.split(args.batchSize, dim=0)

        # Collect results from 
        results_list = [torch.stack(v_list, dim=1).mean(1).cpu() if len(v_list) else torch.zeros(1) for v_list in metric_list]
        args.fps = fps
        h,w = img_data.shape[-2:]
        cur_model.train()

        # Dump predictions and concat into videos
        if dump_vis and args.dump_videos:
            gif_file = os.path.join(args.outf, 'gt_pred' + ('_quant.gif' if model_ind else '.gif'))
            with imageio.get_writer(gif_file, mode='I') as writer:
                for filename in sorted(os.listdir(visual_dir)):
                    image = imageio.v2.imread(os.path.join(visual_dir, filename))
                    writer.append_data(image)
            if not args.dump_images:
                shutil.rmtree(visual_dir)
            # optimize(gif_file)
        
    # dump quantized checkpoint, and decoder
    if local_rank in [0, None] and quant_ckt != None:
        # huffman coding
        if huffman_coding:
            quant_v_list = quant_embed['quant'].flatten().tolist()
            tmin_scale_len = quant_embed['min'].nelement() + quant_embed['scale'].nelement()
            for k, layer_wt in quant_ckt.items():
                quant_v_list.extend(layer_wt['quant'].flatten().tolist())
                tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()

            # get the element name and its frequency
            unique, counts = np.unique(quant_v_list, return_counts=True)
            num_freq = dict(zip(unique, counts))

            # generating HuffmanCoding table
            codec = HuffmanCodec.from_data(quant_v_list)
            sym_bit_dict = {}
            for k, v in codec.get_code_table().items():
                sym_bit_dict[k] = v[0]

            # total bits for quantized embed + model weights
            total_bits = 0
            for num, freq in num_freq.items():
                total_bits += freq * sym_bit_dict[num]
            args.bits_per_param = total_bits / len(quant_v_list)
            
            # including the overhead for min and scale storage, 
            total_bits += tmin_scale_len * 16               #(16bits for float16)
            args.full_bits_per_param = total_bits / len(quant_v_list)

            # bits per pixel
            args.total_bpp = total_bits / args.final_size / args.full_data_length
            print(f'After quantization and encoding: \n bits per parameter: {round(args.full_bits_per_param, 2)}, bits per pixel: {round(args.total_bpp, 4)}')
    # import pdb; pdb.set_trace; from IPython import embed; embed()     

    return results_list, (h,w)


def quant_model(model, args):
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1:
        return model_list, None
    else:
        cur_model = deepcopy(model)
        quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
        encoder_k_list = []
        for k,v in cur_ckt.items():
            if 'encoder' in k:
                encoder_k_list.append(k)
            else:
                quant_v, new_v = quant_tensor(v, args.quant_model_bit)
                quant_ckt[k] = quant_v
                cur_ckt[k] = new_v
        for encoder_k in encoder_k_list:
            del quant_ckt[encoder_k]
        cur_model.load_state_dict(cur_ckt)
        model_list.append(cur_model)
        
        return model_list, quant_ckt


if __name__ == '__main__':
    main()
