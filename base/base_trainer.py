import torch, os, datetime
from timm.utils import NativeScaler

class BaseTrainer:
    def __init__(self, opt, model, optimizer, scheduler, criterion, loss_scaler, logger, writer, model_dir, train_loader, val_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.logger = logger
        self.writer = writer
        self.model_dir = model_dir
        self.opt = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_psnr = 0
        self.best_epoch = 0
        self.best_iter = 0
        self.loss_scaler = loss_scaler
        self.device = device
    def validata_before_train(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def validate(self, epoch):
        raise NotImplementedError("This method should be overridden by subclasses")

    def train_one_epoch(self, epoch, eval_now):
        raise NotImplementedError("This method should be overridden by subclasses")

    def save_best_model(self, psnr_val, epoch, idx):
        self.best_psnr = psnr_val
        self.best_epoch = epoch
        self.best_iter = idx
        torch.save({'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                    os.path.join(self.model_dir, 'model_best.pth'))

    def save_model(self, epoch):
        torch.save({'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.model_dir, 'model_latest.pth'))
        if epoch % self.opt.checkpoint == 0:
            torch.save({'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()},
                       os.path.join(self.model_dir, f'model_epoch_{epoch}.pth'))

    def train(self, start_epoch):
        with torch.no_grad():
            self.validata_before_train()
        self.logger.info('===> Start Epoch {} End Epoch {}'.format(start_epoch, self.opt.nepoch))
        eval_now = len(self.train_loader) // 4
        self.logger.info("Evaluation every {} iterations".format(eval_now))
        torch.cuda.empty_cache()

        for epoch in range(start_epoch, self.opt.nepoch + 1):
            self.train_one_epoch(epoch, eval_now)

        self.logger.info(f"Training is done! and Now time is : {datetime.datetime.now().isoformat()}")
