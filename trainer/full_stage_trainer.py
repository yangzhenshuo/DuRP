import utils
import torch, time
from tqdm import tqdm
from base.base_trainer import BaseTrainer
from utils.model_utils import load_checkpoint

class FullStageTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, scheduler, criterion, loss_scaler, logger, writer, model_dir, train_loader, val_loader, device):
        super(FullStageTrainer, self).__init__(opt, model, optimizer, scheduler, criterion, loss_scaler, logger, writer, model_dir, train_loader, val_loader, device)
        if not opt.resume:
            load_checkpoint(self.model.polModel, opt.PRNet_weights)
            load_checkpoint(self.model.IRModel, opt.IRNet_weights)
            logger.info("Load PRNet weights from: {} and IRNet weights from: {}".format(opt.PRNet_weights, opt.IRNet_weights))
            logger.info("load weights successfully!")
        self.model.cuda(self.device) 
        
    def _calc_unpolaried_img(delf, I_alpha):
        I0, I45, I90, I135 = torch.split(I_alpha, 3, dim=1)
        I_un = (I0 + I45 + I90 + I135)/4
        return I_un
    
    def validata_before_train(self):
        self.model.eval()
        psnr_dataset_s0 = []
        psnr_model_s0 = []
        psnr_dataset_dolp = []
        psnr_model_dolp = []
        val_loss = 0
        total_samples = 0  
        
        for ii, data_val in enumerate((self.val_loader), 0):
            I_alpha = data_val[0].cuda(self.device)
            m_I = data_val[1].cuda(self.device)
            m_D = data_val[2].cuda(self.device)
            m_A = data_val[3].cuda(self.device)
            R = data_val[4].cuda(self.device)
            with torch.amp.autocast('cuda'):
                mA_hat, mD_hat, AInf_hat, R_hat = self.model(I_alpha, m_I)
            # calculate PSNR
            I_un = self._calc_unpolaried_img(I_alpha)   
            psnr_dataset_s0.append(utils.batch_PSNR(R, I_un, False).item())
            psnr_model_s0.append(utils.batch_PSNR(R, R_hat, False).item())
            psnr_dataset_dolp.append(utils.batch_PSNR(m_D, m_I, False).item())
            psnr_model_dolp.append(utils.batch_PSNR(m_D, mD_hat, False).item())
        psnr_dataset_s0 = sum(psnr_dataset_s0)/len(self.val_loader.dataset)
        psnr_model_s0 = sum(psnr_model_s0)/len(self.val_loader.dataset)
        psnr_dataset_dolp = sum(psnr_dataset_dolp)/len(self.val_loader.dataset)
        psnr_model_dolp = sum(psnr_model_dolp)/len(self.val_loader.dataset)
        self.best_psnr = psnr_model_s0
        # print PSNR
        self.logger.info('Input & GT (PSNR S0) --> %.4f dB, (PSNR DoLP) --> %.4f dB' % (psnr_dataset_s0, psnr_dataset_dolp))
        self.logger.info('Model_init & GT (PSNR S0) --> %.4f dB, (PSNR DoLP) --> %.4f dB' % (psnr_model_s0, psnr_model_dolp))
        self.model.train()
        
    def validata(self, epoch):
        self.model.eval()
        psnr_s0 = []
        psnr_dolp = []
        val_loss = 0
        total_samples = 0  # add sample count
        
        for ii, data_val in enumerate((self.val_loader), 0):
            I_alpha = data_val[0].cuda(self.device)
            m_I = data_val[1].cuda(self.device)
            m_D = data_val[2].cuda(self.device)
            m_A = data_val[3].cuda(self.device)
            R = data_val[4].cuda(self.device)
            A_inf = data_val[5].cuda(self.device)
            
            batch_size = I_alpha.size(0)  # get current batch size
            total_samples += batch_size  # accumulate sample count
            with torch.amp.autocast('cuda'):
                mA_hat, mD_hat, AInf_hat, R_hat = self.model(I_alpha, m_I)
                loss = self.criterion(m_A, mA_hat, m_D, mD_hat, A_inf, AInf_hat, R, R_hat)
            val_loss += loss.item() * batch_size  # multiply by batch size and accumulate
            psnr_s0.append(utils.batch_PSNR(R, R_hat, False).item())
            psnr_dolp.append(utils.batch_PSNR(m_D, mD_hat, False).item())
            # self.writer.write_image(R_hat, R, epoch * len(self.val_loader) + ii, mode='val_iter')
        val_loss = val_loss / total_samples if total_samples > 0 else 0
        psnr_s0 = sum(psnr_s0) / len(self.val_loader.dataset)
        psnr_dolp = sum(psnr_dolp) / len(self.val_loader.dataset)
        return psnr_s0, psnr_dolp, val_loss
    
    def train_one_epoch(self, epoch, eval_now):
        epoch_start_time = time.time()
        epoch_loss = 0
        total_samples = 0
        
        for idx, data in enumerate(tqdm(self.train_loader), 0):
            self.optimizer.zero_grad()
            I_alpha = data[0].cuda(self.device)
            m_I = data[1].cuda(self.device)
            m_D = data[2].cuda(self.device)
            m_A = data[3].cuda(self.device)
            R = data[4].cuda(self.device)
            A_inf = data[5].cuda(self.device)
            
            batch_size = I_alpha.size(0)  
            total_samples += batch_size  
            with torch.amp.autocast('cuda'):
                mA_hat, mD_hat, AInf_hat, R_hat = self.model(I_alpha, m_I)
                loss = self.criterion(m_A, mA_hat, m_D, mD_hat, A_inf, AInf_hat, R, R_hat)
            self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters())
            epoch_loss += loss.item() * batch_size

            with torch.no_grad():
                self.writer.write_data({'loss': loss.item()}, epoch * len(self.train_loader) + idx, mode='train_iter')
                if idx % 100 == 0:
                    self.writer.write_image(R_hat, R, epoch * len(self.train_loader) + idx, mode='train_iter')
                if (idx + 1) % eval_now == 0 and idx > 0:
                    psnr_s0, psnr_dolp, val_loss = self.validata(epoch)
                    val_step = (idx + 1) / eval_now
                    self.writer.write_data({'psnr_val_s0': psnr_s0}, epoch * val_step, mode='val_iter')
                    self.writer.write_data({'psnr_val_dolp': psnr_dolp}, epoch * val_step, mode='val_iter')
                    self.writer.write_data({'val_loss': val_loss}, epoch * val_step, mode='val_iter')
                    if psnr_s0 > self.best_psnr:
                        self.save_best_model(psnr_s0, epoch, idx)
                    self.logger.info(
                        "[Epoch %d iteration %d\t S0_PSNR: %.4f dB\t dolp_PSNR: %.4f dB\t ValLoss: %.4f] [Best epoch: %d best_iteration %d Best_PSNR %.4fdB]" %
                        (epoch, idx, psnr_s0, psnr_dolp, val_loss, self.best_epoch, self.best_iter, self.best_psnr))
                    self.model.train()
                    torch.cuda.empty_cache()
                    
        epoch_loss = epoch_loss / total_samples if total_samples > 0 else 0
        self.scheduler.step()
        self.logger.info("Epoch: {}\tTime: {:.4f}\tTrainLoss: {:.6f}\tLearningRate {:.6f}".format(
            epoch, time.time() - epoch_start_time, epoch_loss, self.scheduler.get_last_lr()[0]))

        self.save_model(epoch)
