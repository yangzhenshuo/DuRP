import argparse

class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, description=''):
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--description', type=str, default='full stage', help='Description for the run')        
        # global settings
        parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        parser.add_argument('--nepoch', type=int, default=400, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=8, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=8, help='eval_dataloader workers')
        parser.add_argument('--optimizer', type=str, default ='adam', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0001, help='initial learning rate')
        parser.add_argument('--T_max', type=int, default=32, help='T_max for CosineAnnealingLR')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
        parser.add_argument('--seed', type=int, default=10, help='random seed')
        parser.add_argument('--eta_min', type=float, default=1e-5, help='min_eta')

        # args for stage
        parser.add_argument('--loss', type=str, default ='pol_loss', choices = ['pol_loss', 'img_loss', 'full_loss'], help='loss function')
        parser.add_argument('--arch', type=str, default ='PRNet', choices = ['PRNet', 'IRNet', 'PENN'], help='archtechture')
        parser.add_argument('--stage', type=str, default='pol', choices = ['pol', 'img', 'full'], help='stage of training')

        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='./experiment/',  help='save dir')
        parser.add_argument('--env', type=str, default ='_0',  help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')
        parser.add_argument('--log_name', type=str, default ='train',  help='log name')
        
        # args for training
        parser.add_argument('--train_ps', type=int, default=512, help='patch size of training sample')
        parser.add_argument('--val_ps', type=int, default=512, help='patch size of validation sample')
        parser.add_argument('--resume', action='store_true',default=False, help='resume training')
        parser.add_argument('--pretrain_weights',type=str, default='', help='path of pretrained_weights')
        parser.add_argument('--train_dir', type=str, default ='./datas/SRPG/train',  help='dir of train data')
        parser.add_argument('--val_dir', type=str, default ='./datas/SRPG/test',  help='dir of test data')
        parser.add_argument('--PRNet_weights', type=str, default ='',  help='PRNet weights')
        parser.add_argument('--IRNet_weights', type=str, default ='',  help='IRNet weights')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup') 
        parser.add_argument('--gpu_device', type=str, default='0', help='GPU id')
        # ddp
        parser.add_argument("--local_rank", type=int,default=-1,help='DDP parameter, do not modify')
        parser.add_argument("--distribute",action='store_true',help='whether using multi gpu train')
        parser.add_argument("--distribute_mode",type=str,default='DDP',help="using which mode to ")
        return parser
