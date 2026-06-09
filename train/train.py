import os, datetime, options, random, sys
# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
import utils
import logging
import torch
import torch.optim as optim
import numpy as np
from scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from losses import get_loss
from dataset import get_training_data, get_validation_data
from torch.utils.data import DataLoader
from timm.utils import NativeScaler
from trainer.img_stage_trainer import ImgStageTrainer
from trainer.pol_stage_trainer import PolStageTrainer
from trainer.full_stage_trainer import FullStageTrainer

def set_seeds(seed=3407):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def logger_tensorboard_init(opt):
    log_dir = os.path.join(opt.save_dir, opt.arch+opt.env, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.log')
    logger = utils.get_root_logger(logger_name=opt.log_name, log_level=logging.INFO, log_file=log_file)
    write_dir = os.path.join(opt.save_dir, opt.arch+opt.env, 'runs')
    write = utils.WriterTensorboardX(write_dir, logger, True)
    return logger, write

def get_trainer(opt, model, optimizer, scheduler, criterion, loss_scaler, logger, writer, model_dir, train_loader, val_loader, device):
    arch = opt.arch
    logger.info("Using model: {}".format(arch))
    if arch == 'PRNet':
        trainer = PolStageTrainer(opt, model, optimizer, scheduler, criterion, loss_scaler, logger, writer, model_dir, train_loader, val_loader, device)
    elif arch == 'IRNet':
        trainer = ImgStageTrainer(opt, model, optimizer, scheduler, criterion, loss_scaler, logger, writer, model_dir, train_loader, val_loader, device)
    elif arch == 'PENN':
        trainer = FullStageTrainer(opt, model, optimizer, scheduler, criterion, loss_scaler, logger, writer, model_dir, train_loader, val_loader, device)
    else:    
        raise Exception("Arch error!")
    return trainer
######### parser ###########
opt = options.Options().init().parse_args()
print(f"Description: {opt.description}")
######### Logs and Tensorboard ###########
logger, writer = logger_tensorboard_init(opt)
logger.info("Now time is : {}".format(datetime.datetime.now().isoformat()))

######### print options ###########
logger.info(utils.get_env_info())
logger.info(utils.dict2str(opt))

######### Checkpoint ############
model_dir = os.path.join(opt.save_dir, opt.arch+opt.env, 'checkpoints')
utils.mkdir(model_dir)

########## Set Seeds ###########
set_seeds(opt.seed)

########## Model ###########
model = utils.get_arch(opt, logger)
logger.info(str(model))
logger.info('Trainable parameters: {:.2f}M'.format(model.summary()/1e6))
logger.info('Training stage: {}'.format(opt.stage))
######### Optimizer ###########
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(trainable_params, lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(trainable_params, lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### DataParallel ########### 
if torch.cuda.device_count() > 1 and opt.distribute:
    gpu_ids = list(map(int, opt.gpu_device.split(',')))  
    device = torch.device("cuda:{}".format(gpu_ids[0]))
    model.cuda(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
else:
    device = torch.device("cuda:{}".format(opt.gpu_device) if torch.cuda.is_available() else "cpu")
    model.cuda(device)
######### Scheduler ###########
if opt.warmup:
    logger.info("Using warmup and cosine strategy")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
else:
    # scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    T_max = opt.T_max
    eta_min = opt.eta_min
    logger.info("Using CosineAnnealingLR, T_max={} and eta_min={}!".format(T_max, eta_min))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
######### Resume ########### 
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    logger.info("loading checkpoint from {}".format(path_chk_rest))
    utils.load_checkpoint(model, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)
    for idx in range(1, start_epoch):
        scheduler.step()
    new_lr = optimizer.param_groups[0]['lr']
    logger.info("resume from epoch {}, lr is {}".format(start_epoch-1, new_lr))
    
######### Loss ###########
criterion = get_loss(opt).cuda(device)

######### DataLoader ###########
logger.info('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=False, drop_last=True)

val_dataset = get_validation_data(opt)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
logger.info('===> Training dataset length: {} and Validation dataset length: {}'\
            .format(len_trainset, len_valset))

########## Trainer ###########
loss_scaler = NativeScaler()
trainer = get_trainer(opt, model, optimizer, scheduler, criterion, loss_scaler, logger, writer, model_dir, train_loader, val_loader, device)
########## Set default dtype ###########
torch.set_default_dtype(torch.float32)
########## Training ###########
trainer.train(start_epoch)
