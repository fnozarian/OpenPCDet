import argparse
import datetime
import glob
import os
import copy
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_semi_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.semi_train_utils import train_ssl_model
from train_utils.train_utils import train_model
from eval_utils.eval_utils import eval_one_epoch
import subprocess

from pcdet.ssod.sess import sess
from pcdet.ssod.pseudo_label import pseudo_label
from pcdet.ssod.iou_match_3d import iou_match_3d
from pcdet.ssod.se_ssd import se_ssd
from pcdet.ssod.contrastive import Contrastive

# TODO(farzad): Make classes for each semi learning methods similar to Contrastive
semi_learning_models = {
    'SESS': sess,
    'Pseudo-Label': pseudo_label,
    '3DIoUMatch': iou_match_3d,
    'SE_SSD': se_ssd,
    'Contrastive': Contrastive
}


def get_git_commit_number():
    if not os.path.exists('../.git'):
        return '0000000'
    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--eval_batch_size', type=int, default=None, required=False, help='batch size for testing')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=5, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--split', type=str, default='train_0.01_1')
    parser.add_argument('--thresh', type=str, default='0.5, 0.25, 0.25')
    parser.add_argument('--sem_thresh', type=str, default='0.4, 0.4, 0.4')
    parser.add_argument('--unlabeled_weight', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    assert cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NAME == 'gt_sampling'  # hardcode
    cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].DB_INFO_PATH = [f'kitti_dbinfos_{args.split}.pkl']
    cfg.DATA_CONFIG.DATA_SPLIT['train'] = args.split
    cfg.MODEL.THRESH = [float(x) for x in args.thresh.split(',')]
    cfg.MODEL.SEM_THRESH = [float(x) for x in args.sem_thresh.split(',')]
    cfg.MODEL.UNLABELED_WEIGHT = args.unlabeled_weight

    rev = get_git_commit_number()
    args.extra_tag = args.extra_tag + "_" + str(rev)

    if args.lr > 0.0:
        cfg.OPTIMIZATION.LR = args.lr

    # cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.score_thresh

    return args, cfg


def main():

    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    pretrain_ckpt_dir = output_dir / 'pretrain_ckpt'
    ssl_ckpt_dir = output_dir / 'ssl_ckpt'
    student_ckpt_dir = output_dir / 'ssl_ckpt' / 'student'
    teacher_ckpt_dir = output_dir / 'ssl_ckpt' / 'teacher'

    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_ckpt_dir.mkdir(parents=True, exist_ok=True)
    student_ckpt_dir.mkdir(parents=True, exist_ok=True)
    teacher_ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        lbl_bs = int(cfg.OPTIMIZATION.SEMI_SUP_LEARNING.LD_BATCH_SIZE_PER_GPU)
        ulb_bs = int(cfg.OPTIMIZATION.SEMI_SUP_LEARNING.UD_BATCH_SIZE_PER_GPU)
        total_bs = lbl_bs + ulb_bs
        logger.info('total_batch_size: %d' % (total_gpus * total_bs))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))
        tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
    else:
        tb_log = None

    # -----------------------create dataloader & network & optimizer---------------------------
    batch_size = {
        'pretrain': cfg.OPTIMIZATION.PRETRAIN.BATCH_SIZE_PER_GPU,
        'labeled': cfg.OPTIMIZATION.SEMI_SUP_LEARNING.LD_BATCH_SIZE_PER_GPU,
        'unlabeled': cfg.OPTIMIZATION.SEMI_SUP_LEARNING.UD_BATCH_SIZE_PER_GPU,
        'test': cfg.OPTIMIZATION.TEST.BATCH_SIZE_PER_GPU,
    }
    repeat = {
        'pretrain': cfg.OPTIMIZATION.PRETRAIN.REPEAT,
        'labeled': cfg.OPTIMIZATION.SEMI_SUP_LEARNING.REPEAT,
        'unlabeled': 1,
        'test': 1
    }
    # -----------------------create dataloader & network & optimizer---------------------------
    datasets, dataloaders, samplers = build_semi_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        repeat=repeat,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        seed=666 if args.fix_random_seed else None
    )
    # --------------------------------stage I pretraining---------------------------------------
    logger.info('************************Stage I Pretraining************************')
    model_cfgs = copy.deepcopy(cfg.MODEL)
    pretrain_model = build_network(model_cfg=model_cfgs, num_class=len(cfg.CLASS_NAMES),
                                   dataset=datasets['pretrain'])

    if args.pretrained_model is not None:
        eval_pretrain_dir = output_dir / 'eval' / 'eval_with_pretraining'
        # skip eval if there is a result.pkl is found in the eval_pretrain_dir
        if not (eval_pretrain_dir / 'result.pkl').exists():
            pretrain_ckpt = args.pretrained_model

            pretrain_model.load_params_from_file(filename=pretrain_ckpt, logger=logger, to_cpu=dist_train)
            pretrain_model.cuda()
            pretrain_model.eval()  # before wrap to DistributedDataParallel to support fixed some parameters
            if dist_train:
                pretrain_model = nn.parallel.DistributedDataParallel(pretrain_model, device_ids=[
                    cfg.LOCAL_RANK % torch.cuda.device_count()])
            logger.info(pretrain_model)
            eval_pretrain_dir.mkdir(parents=True, exist_ok=True)
            # eval_one_epoch(cfg, pretrain_model, dataloaders['test'], -1, logger, dist_test=dist_train,
            #                save_to_file=False, result_dir=eval_pretrain_dir)
    else:
        pretrain_model.cuda()
        pretrain_optimizer = build_optimizer(pretrain_model, cfg.OPTIMIZATION.PRETRAIN)

        pretrain_model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
        if dist_train:
            pretrain_model = nn.parallel.DistributedDataParallel(pretrain_model, device_ids=[
                cfg.LOCAL_RANK % torch.cuda.device_count()])
        logger.info(pretrain_model)
        last_epoch = -1
        start_epoch = it = 0
        pretrain_lr_scheduler, pretrain_lr_warmup_scheduler = build_scheduler(
            pretrain_optimizer, total_iters_each_epoch=len(dataloaders['pretrain']),
            total_epochs=cfg.OPTIMIZATION.PRETRAIN.NUM_EPOCHS,
            last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION.PRETRAIN
        )
        logger.info('**********************Start pre-training %s/%s/%s)**********************'
                    % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
        train_model(
            pretrain_model,
            pretrain_optimizer,
            dataloaders['pretrain'],
            model_func=model_fn_decorator(),
            lr_scheduler=pretrain_lr_scheduler,
            optim_cfg=cfg.OPTIMIZATION.PRETRAIN,
            start_epoch=start_epoch,
            total_epochs=cfg.OPTIMIZATION.PRETRAIN.NUM_EPOCHS,
            start_iter=it,
            rank=cfg.LOCAL_RANK,
            tb_log=tb_log,
            ckpt_save_dir=pretrain_ckpt_dir,
            train_sampler=samplers['pretrain'],
            lr_warmup_scheduler=pretrain_lr_warmup_scheduler,
            ckpt_save_interval=args.ckpt_save_interval,
            max_ckpt_save_num=args.max_ckpt_save_num,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
        )
        logger.info('**********************End pre-training %s/%s/%s)**********************\n\n\n'
                    % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

        logger.info('**********************Start evaluation for pre-training %s/%s/%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
        eval_pretrain_dir = output_dir / 'eval' / 'eval_with_pretraining'
        eval_pretrain_dir.mkdir(parents=True, exist_ok=True)
        args.start_epoch = cfg.OPTIMIZATION.PRETRAIN.NUM_EPOCHS - 10
        repeat_eval_ckpt(
            model=pretrain_model.module if dist_train else pretrain_model,
            test_loader=dataloaders['test'],
            args=args,
            eval_output_dir=eval_pretrain_dir,
            logger=logger,
            ckpt_dir=pretrain_ckpt_dir,
            dist_test=dist_train
        )
        logger.info('**********************End evaluation for pre-training %s/%s/%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # --------------------------------stage II SSL training---------------------------------------
    logger.info('************************Stage II SSL training************************')
    model_name = cfg.OPTIMIZATION.SEMI_SUP_LEARNING.NAME
    model = semi_learning_models[model_name](cfg, datasets)
    model.cuda()

    # load checkpoint if it is possible
    last_epoch = -1
    start_epoch = it = 0
    based_on_pretrained = True
    teacher_ckpt_list = glob.glob(str(teacher_ckpt_dir / '*checkpoint_epoch_*.pth'))
    student_ckpt_list = glob.glob(str(student_ckpt_dir / '*checkpoint_epoch_*.pth'))
    if len(teacher_ckpt_list) > 0 and len(student_ckpt_list) > 0:
        optimizer = build_optimizer(model, cfg.OPTIMIZATION.SEMI_SUP_LEARNING.STUDENT)
        based_on_pretrained = False
        teacher_ckpt_list.sort(key=os.path.getmtime)
        student_ckpt_list.sort(key=os.path.getmtime)
        it, start_epoch = model.teacher.load_params_with_optimizer(
            teacher_ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
        )
        it, start_epoch = model.student.load_params_with_optimizer(
            student_ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch + 1
    else:
        optimizer = build_optimizer(model, cfg.OPTIMIZATION.SEMI_SUP_LEARNING.STUDENT)

    if based_on_pretrained:
        if args.pretrained_model is not None:
            pretrained_model = args.pretrained_model

        else:
            ckpt_list = glob.glob(str(pretrain_ckpt_dir / '*checkpoint_epoch_*.pth'))
            ckpt_list.sort(key=os.path.getmtime)
            pretrained_model = ckpt_list[-1]

        model.teacher.load_params_from_file(filename=pretrained_model, to_cpu=dist_train, logger=logger)
        model.student.load_params_from_file(filename=pretrained_model, to_cpu=dist_train, logger=logger)

    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    
    logger.info(model)
    # use labeled data as epoch counter
    student_lr_scheduler, student_lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(dataloaders['labeled']),
        total_epochs=cfg.OPTIMIZATION.SEMI_SUP_LEARNING.NUM_EPOCHS,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION.SEMI_SUP_LEARNING.STUDENT
    )
    logger.info('**********************Start ssl-training %s/%s/%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_ssl_model(
        model=model,
        optimizer=optimizer,
        labeled_loader=dataloaders['labeled'],
        unlabeled_loader=dataloaders['unlabeled'],
        lr_scheduler=student_lr_scheduler,
        ssl_cfg=cfg.OPTIMIZATION.SEMI_SUP_LEARNING,
        start_epoch=start_epoch,
        total_epochs=cfg.OPTIMIZATION.SEMI_SUP_LEARNING.NUM_EPOCHS,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ssl_ckpt_dir,
        labeled_sampler=samplers['labeled'],
        unlabeled_sampler=samplers['unlabeled'],
        lr_warmup_scheduler=student_lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        dist=dist_train
    )

    logger.info('**********************End ssl-training %s/%s/%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation for student model %s/%s/%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    eval_ssl_dir = output_dir / 'eval' / 'eval_with_student_model'
    eval_ssl_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = cfg.OPTIMIZATION.SEMI_SUP_LEARNING.NUM_EPOCHS - 25
    repeat_eval_ckpt(
        model=model.module.student if dist_train else model.student,
        test_loader=dataloaders['test'],
        args=args,
        eval_output_dir=eval_ssl_dir,
        logger=logger,
        ckpt_dir=ssl_ckpt_dir / 'student',
        dist_test=dist_train
    )
    logger.info('**********************End evaluation for student model %s/%s/%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation for teacher model %s/%s/%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    eval_ssl_dir = output_dir / 'eval' / 'eval_with_teacher_model'
    eval_ssl_dir.mkdir(parents=True, exist_ok=True)

    for t_param in model.module.teacher.parameters(): # Add this to avoid errors
        t_param.requires_grad = True
    args.start_epoch = cfg.OPTIMIZATION.SEMI_SUP_LEARNING.NUM_EPOCHS - 25
    repeat_eval_ckpt(
        model=model.module.teacher if dist_train else model.teacher,
        test_loader=dataloaders['test'],
        args=args,
        eval_output_dir=eval_ssl_dir,
        logger=logger,
        ckpt_dir=ssl_ckpt_dir / 'teacher',
        dist_test=dist_train
    )
    logger.info('**********************End evaluation for teacher model %s/%s/%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
