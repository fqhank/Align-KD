import os
import argparse
from copy import deepcopy
import numpy as np
import tqdm, sys

def run_cmd(args):
    os.system('cd MobileVLM-distill/')
    if args.task == 'pretrain-finetune-test':
        cmd = 'bash run.sh {} {} openai/{} openai/{} {} {}'.format(args.arch,args.task,args.llm,args.v_encoder,args.data_path,args.distill)
    else:
        cmd = 'bash run.sh {} {} openai/{} openai/{} {} {} {}'.format(args.arch,args.task,args.llm,args.v_encoder,args.out_path, args.data_path,args.distill)

    print('='*30)
    print(cmd)
    print('-'*30)
    os.system(cmd)
    print('-'*30)


if __name__ == '__main__':
	parser = argparse.ArgumentParser('MobileLlaMA-Distillation')
	# running script parsers
	parser.add_argument('--data_path',type=str,default='mobilevlmv2_data/')
	parser.add_argument('--pretrained_model_path',type=str, default='checkpoints/')
	parser.add_argument('--num_GPUs',type=int, default=8)
	parser.add_argument('--task',type=str,default='pretrain-finetune-test') # pretrain/finetune/test/pretrain-finetune-test
	parser.add_argument('--arch',type=str,default='mobilevlm_v2_1.7b')
	parser.add_argument('--v_encoder',type=str,default='clip-vit-large-patch14-336')
	parser.add_argument('--llm',type=str,default='MobileLlaMA-1.4B-Chat')
	parser.add_argument('--distill',type=int,default=1)
	parser.add_argument('--out_path',type=str,default='.')

	args, unparsed = parser.parse_known_args()

	# prepare data at here if needed. 
    
	# run cmd
	run_cmd(args)

	print('training ends.')
	print('============================END=============================')