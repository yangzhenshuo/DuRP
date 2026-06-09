import utils
import torch, time
from tqdm import tqdm
from base.base_trainer import BaseTrainer

class PolStageTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, scheduler, criterion, loss_scaler, logger, writer, model_dir, train_loader, val_loader, device):
        super(PolStageTrainer, self).__init__(opt, model, optimizer, scheduler, criterion, loss_scaler, logger, writer, model_dir, train_loader, val_loader, device)
    def validata_before_train(self):
        self.model.eval()
        psnr_dataset = []
        psnr_model_init = []
        for ii, data_val in enumerate((self.val_loader), 0):
            I_alpha = data_val[0].cuda(self.device)
            m_I = data_val[1].cuda(self.device)
            m_D = data_val[2].cuda(self.device)
            K = data_val[4].cuda(self.device)
            K_hat, m_A_hat, m_D_hat = self.model(I_alpha, m_I)    
            psnr_dataset.append(utils.batch_PSNR(m_D, m_I, False).item())
            psnr_model_init.append(utils.batch_PSNR(m_D, m_D_hat, False).item())
        psnr_dataset = sum(psnr_dataset)/len(self.val_loader.dataset)
        psnr_model_init = sum(psnr_model_init)/len(self.val_loader.dataset)
        self.logger.info('Input & GT (PSNR) --> %.4f dB' % psnr_dataset)
        self.logger.info('Model_init & GT (PSNR) --> %.4f dB' % psnr_model_init)
        self.model.train()
        
    def validata(self, epoch):
        self.model.eval()
        psnr_dataset = []
        val_loss = 0
        for ii, data_val in enumerate((self.val_loader), 0):
            I_alpha = data_val[0].cuda(self.device)
            m_I = data_val[1].cuda(self.device)
            m_D = data_val[2].cuda(self.device)
            m_A = data_val[3].cuda(self.device)
            K = data_val[4].cuda(self.device)
            K_hat, m_A_hat, m_D_hat = self.model(I_alpha, m_I) 
            loss = self.criterion(K, K_hat, m_A, m_A_hat, m_D, m_D_hat)
            val_loss += loss.item()
            psnr_dataset.append(utils.batch_PSNR(m_D, m_D_hat, False).item())
            # self.writer.write_image(m_D, m_D_hat, epoch * len(self.val_loader) + ii, mode='val_iter')
        psnr_avg = sum(psnr_dataset) / len(self.val_loader.dataset)
        return psnr_avg, val_loss
    
    def train_one_epoch(self, epoch, eval_now):
        epoch_start_time = time.time()
        epoch_loss = 0
        for idx, data in enumerate(tqdm(self.train_loader), 0):
            self.optimizer.zero_grad()
            I_alpha = data[0].cuda(self.device)
            m_I = data[1].cuda(self.device)
            m_D = data[2].cuda(self.device)
            m_A = data[3].cuda(self.device)
            K = data[4].cuda(self.device)
            K_hat, m_A_hat, m_D_hat = self.model(I_alpha, m_I) 
            loss = self.criterion(K, K_hat, m_A, m_A_hat, m_D, m_D_hat)
            self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters())
            epoch_loss += loss.item()

            with torch.no_grad():
                self.writer.write_data({'loss': loss.item()}, epoch * len(self.train_loader) + idx, mode='train_iter')
                if idx % 100 == 0:
                    self.writer.write_image(m_D, m_D_hat, epoch * len(self.train_loader) + idx, mode='train_iter')
                if (idx + 1) % eval_now == 0 and idx > 0:
                    psnr_val, val_loss = self.validata(epoch)
                    val_step = (idx + 1) / eval_now
                    self.writer.write_data({'psnr_val': psnr_val}, epoch * val_step, mode='val_iter')
                    self.writer.write_data({'val_loss': val_loss}, epoch * val_step, mode='val_iter')
                    if psnr_val > self.best_psnr:
                        self.save_best_model(psnr_val, epoch, idx)
                    self.logger.info(
                        "[Epoch %d iteration %d\t PSNR: %.4f dB\t ValLoss: %.4f] [Best epoch: %d best_iteration %d Best_PSNR %.4fdB]" %
                        (epoch, idx, psnr_val, val_loss, self.best_epoch, self.best_iter, self.best_psnr))
                    self.model.train()
                    torch.cuda.empty_cache()

        self.scheduler.step()
        self.logger.info("Epoch: {}\tTime: {:.4f}\tTrainLoss: {:.6f}\tLearningRate {:.6f}".format(
            epoch, time.time() - epoch_start_time, epoch_loss, self.scheduler.get_last_lr()[0]))

        self.save_model(epoch)
