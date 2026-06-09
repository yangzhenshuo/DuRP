import numpy as np
import os, fnmatch
from torch.utils.data import Dataset
import torch
from utils import Augment_RGB_torch
import random
augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
##################################################################################################
class DataLoader_pol(Dataset):
    def __init__(self, npy_dir, is_train=True, img_options=None, target_transform=None):
        """
        polarization recovery stage dataset
        
        args:
            npy_dir (str): dataset directory
            is_train (bool): is training mode, default is True
            img_options (dict): image options, such as patch size
            target_transform (bool): whether to apply data augmentation
        """
        super(DataLoader_pol, self).__init__()

        self.is_train = is_train
        self.target_transform = target_transform
        self.input_dir = os.path.join(npy_dir, 'input')
        self.target_dir = os.path.join(npy_dir, 'target')
        
        self.I_alpha_dir = os.path.join(self.input_dir, 'I_alpha')
        self.mI_dir = os.path.join(self.input_dir, 'm_I')
        
        self.mD_dir = os.path.join(self.target_dir, 'm_D')
        self.mA_dir = os.path.join(self.target_dir, 'm_A')
        
        self.k_dir = os.path.join(self.target_dir, 'K')
        self.npy_filenames = fnmatch.filter(os.listdir(self.I_alpha_dir), '*.npy')
        
        if self.is_train:
            self.img_options = img_options
        
        self.tar_size = len(self.npy_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I_alpha = np.load(os.path.join(self.I_alpha_dir, self.npy_filenames[index]))
        m_I = np.load(os.path.join(self.mI_dir, self.npy_filenames[index]))
        m_D = np.load(os.path.join(self.mD_dir, self.npy_filenames[index]))
        m_A = np.load(os.path.join(self.mA_dir, self.npy_filenames[index]))
        K = np.load(os.path.join(self.k_dir, self.npy_filenames[index]))
        
        # share the same memory
        I_alpha = torch.from_numpy(np.float32(I_alpha))
        m_I = torch.from_numpy(np.complex64(m_I))
        m_D = torch.from_numpy(np.complex64(m_D))
        m_A = torch.from_numpy(np.complex64(m_A))
        K = torch.from_numpy(np.float32(K))
        
        I_alpha = I_alpha.permute(2,0,1)
        m_I = m_I.permute(2,0,1)
        m_D = m_D.permute(2,0,1)
        m_A = m_A.permute(2,0,1)
        K = K.permute(2,0,1)
        
        # Training mode specific processing (optional cropping operation, commented out)
        if self.is_train:
            # The following is the original cropping code, kept in comment status
            ps = self.img_options['patch_size']
            H = I_alpha.shape[1]
            W = I_alpha.shape[2]
            if H-ps==0:
                r=0
                c=0
            else:
                r = np.random.randint(0, H - ps)
                c = np.random.randint(0, W - ps)
            I_alpha = I_alpha[:, r:r + ps, c:c + ps]
            m_I = m_I[:, r:r + ps, c:c + ps]
            m_D = m_D[:, r:r + ps, c:c + ps]
            m_A = m_A[:, r:r + ps, c:c + ps]
            K = K[:, r:r + ps, c:c + ps]
            
        name = self.npy_filenames[index].split('.')[0]
        
        # Apply data augmentation (if enabled)
        if self.target_transform:
            apply_trans = transforms_aug[random.getrandbits(3)]
            I_alpha = getattr(augment, apply_trans)(I_alpha)
            m_I = getattr(augment, apply_trans)(m_I)
            m_D = getattr(augment, apply_trans)(m_D)        
            m_A = getattr(augment, apply_trans)(m_A)
            K = getattr(augment, apply_trans)(K)
            
        return I_alpha, m_I, m_D, m_A, K, name
##################################################################################################
class DataLoader_img(Dataset):
    def __init__(self, npy_dir, is_train=True, img_options=None, target_transform=None):
        """
        image recovery stage dataset
        args:
            npy_dir (str): dataset directory
            is_train (bool): is training mode, default is True
            img_options (dict): image options, such as patch size
            target_transform (bool): whether to apply data augmentation
        """
        super(DataLoader_img, self).__init__()

        self.is_train = is_train
        self.target_transform = target_transform
        self.input_dir = os.path.join(npy_dir, 'input')
        self.target_dir = os.path.join(npy_dir, 'target')
        
        self.I_alpha_dir = os.path.join(self.input_dir, 'I_alpha')
        self.mI_dir = os.path.join(self.input_dir, 'm_I')
        
        self.mD_dir = os.path.join(self.target_dir, 'm_D')
        self.mA_dir = os.path.join(self.target_dir, 'm_A')
        
        self.R_dir = os.path.join(self.target_dir, 'R')
        self.AInf_dir = os.path.join(self.target_dir, 'A_inf')
        self.npy_filenames = fnmatch.filter(os.listdir(self.I_alpha_dir), '*.npy')
        
        if self.is_train:
            self.img_options = img_options
            
        self.tar_size = len(self.npy_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I_alpha = np.load(os.path.join(self.I_alpha_dir, self.npy_filenames[index]))
        m_I = np.load(os.path.join(self.mI_dir, self.npy_filenames[index]))
        m_D = np.load(os.path.join(self.mD_dir, self.npy_filenames[index]))
        m_A = np.load(os.path.join(self.mA_dir, self.npy_filenames[index]))
        R = np.load(os.path.join(self.R_dir, self.npy_filenames[index]))
        A_inf = np.load(os.path.join(self.AInf_dir, self.npy_filenames[index]))
        
        # share the same memory
        I_alpha = torch.from_numpy(np.float32(I_alpha))
        m_I = torch.from_numpy(np.complex64(m_I))
        m_D = torch.from_numpy(np.complex64(m_D))
        m_A = torch.from_numpy(np.complex64(m_A))
        R = torch.from_numpy(np.float32(R))
        A_inf = torch.from_numpy(np.float32(A_inf))
        
        I_alpha = I_alpha.permute(2,0,1)
        m_I = m_I.permute(2,0,1)
        m_D = m_D.permute(2,0,1)
        m_A = m_A.permute(2,0,1)
        R = R.permute(2,0,1)
        A_inf = A_inf.permute(2,0,1)
        
        # Training mode specific processing (optional cropping operation, commented out)
        if self.is_train:
            # The following is the original cropping code, kept in comment status
            ps = self.img_options['patch_size']
            H = I_alpha.shape[1]
            W = I_alpha.shape[2]
            if H-ps==0:
                r=0
                c=0
            else:
                r = np.random.randint(0, H - ps)
                c = np.random.randint(0, W - ps)
            I_alpha = I_alpha[:, r:r + ps, c:c + ps]
            m_I = m_I[:, r:r + ps, c:c + ps]
            m_D = m_D[:, r:r + ps, c:c + ps]
            m_A = m_A[:, r:r + ps, c:c + ps]
            R = R[:, r:r + ps, c:c + ps]
            A_inf = A_inf[:, r:r + ps, c:c + ps]
            
        name = self.npy_filenames[index].split('.')[0]
        
        # Apply data augmentation in training mode only (if enabled)
        if self.is_train and self.target_transform:
            apply_trans = transforms_aug[random.getrandbits(3)]
            I_alpha = getattr(augment, apply_trans)(I_alpha)
            m_I = getattr(augment, apply_trans)(m_I)
            m_D = getattr(augment, apply_trans)(m_D)        
            m_A = getattr(augment, apply_trans)(m_A)
            R = getattr(augment, apply_trans)(R)
            
        return I_alpha, m_I, m_D, m_A, R, A_inf, name
##################################################################################################
class DataLoader_full(Dataset):
    def __init__(self, npy_dir, is_train=True, img_options=None, target_transform=None):
        """
        full model dataset
        args:
            npy_dir (str): dataset directory
            is_train (bool): is training mode, default is True
            img_options (dict): image options, such as patch size
            target_transform (bool): whether to apply data augmentation
        """
        super(DataLoader_full, self).__init__()

        self.is_train = is_train
        self.target_transform = target_transform
        self.input_dir = os.path.join(npy_dir, 'input')
        self.target_dir = os.path.join(npy_dir, 'target')
        
        self.I_alpha_dir = os.path.join(self.input_dir, 'I_alpha')
        self.mI_dir = os.path.join(self.input_dir, 'm_I')
        
        self.mD_dir = os.path.join(self.target_dir, 'm_D')
        self.mA_dir = os.path.join(self.target_dir, 'm_A')
        self.R_dir = os.path.join(self.target_dir, 'R')
        self.AInf_dir = os.path.join(self.target_dir, 'A_inf')
        self.npy_filenames = fnmatch.filter(os.listdir(self.I_alpha_dir), '*.npy')
        
        if self.is_train:
            self.img_options = img_options
            
        self.tar_size = len(self.npy_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I_alpha = np.load(os.path.join(self.I_alpha_dir, self.npy_filenames[index]))
        m_I = np.load(os.path.join(self.mI_dir, self.npy_filenames[index]))
        m_D = np.load(os.path.join(self.mD_dir, self.npy_filenames[index]))
        m_A = np.load(os.path.join(self.mA_dir, self.npy_filenames[index]))
        R = np.load(os.path.join(self.R_dir, self.npy_filenames[index]))
        A_inf = np.load(os.path.join(self.AInf_dir, self.npy_filenames[index]))
        
        # share the same memory
        I_alpha = torch.from_numpy(np.float32(I_alpha))
        m_I = torch.from_numpy(np.complex64(m_I))
        m_D = torch.from_numpy(np.complex64(m_D))
        m_A = torch.from_numpy(np.complex64(m_A))
        R = torch.from_numpy(np.float32(R))
        A_inf = torch.from_numpy(np.float32(A_inf))
        
        I_alpha = I_alpha.permute(2,0,1)
        m_I = m_I.permute(2,0,1)
        m_D = m_D.permute(2,0,1)
        m_A = m_A.permute(2,0,1)
        R = R.permute(2,0,1)
        A_inf = A_inf.permute(2,0,1)
        
        # Training mode specific processing (optional cropping operation, commented out)
        if self.is_train:
            # The following is the original cropping code, kept in comment status
            ps = self.img_options['patch_size']
            H = I_alpha.shape[1]
            W = I_alpha.shape[2]
            if H-ps==0:
                r=0
                c=0
            else:
                r = np.random.randint(0, H - ps)
                c = np.random.randint(0, W - ps)
            I_alpha = I_alpha[:, r:r + ps, c:c + ps]
            m_I = m_I[:, r:r + ps, c:c + ps]
            m_D = m_D[:, r:r + ps, c:c + ps]
            m_A = m_A[:, r:r + ps, c:c + ps]
            R = R[:, r:r + ps, c:c + ps]
            A_inf = A_inf[:, r:r + ps, c:c + ps]
            
        name = self.npy_filenames[index].split('.')[0]
        
        # Apply data augmentation in training mode only (if enabled)
        if self.is_train and self.target_transform:
            apply_trans = transforms_aug[random.getrandbits(3)]
            I_alpha = getattr(augment, apply_trans)(I_alpha)
            m_I = getattr(augment, apply_trans)(m_I)
            m_D = getattr(augment, apply_trans)(m_D)        
            m_A = getattr(augment, apply_trans)(m_A)
            R = getattr(augment, apply_trans)(R)
            
        return I_alpha, m_I, m_D, m_A, R, A_inf, name
##################################################################################################
class DataLoader_test(Dataset):
    """
    Test dataset
    
    args:
        npy_dir (str): dataset directory
    """
    def __init__(self, npy_dir, target_size=(512, 512)):
        super(DataLoader_test, self).__init__()
        self.input_dir = os.path.join(npy_dir, 'input')
        self.I_alpha_dir = os.path.join(self.input_dir, 'I_alpha')
        self.mI_dir = os.path.join(self.input_dir, 'm_I')
        self.npy_filenames = fnmatch.filter(os.listdir(self.I_alpha_dir), '*.npy')
        self.tar_size = len(self.npy_filenames)  # get the size of target
        self.target_size = target_size  # Target size, default to 512x512

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I_alpha = np.load(os.path.join(self.I_alpha_dir, self.npy_filenames[index]))
        m_I = np.load(os.path.join(self.mI_dir, self.npy_filenames[index]))
        # share the same meomory
        I_alpha = torch.from_numpy(np.float32(I_alpha))
        m_I = torch.from_numpy(np.complex64(m_I))
    
        I_alpha = I_alpha.permute(2,0,1)
        m_I = m_I.permute(2,0,1)
        
        # Check if size is target size, if not perform bicubic interpolation
        current_h, current_w = I_alpha.shape[1], I_alpha.shape[2]
        if current_h != self.target_size[0] or current_w != self.target_size[1]:
            # Use bicubic interpolation to adjust I_alpha size
            I_alpha = torch.nn.functional.interpolate(
                I_alpha.unsqueeze(0),  # Add batch dimension
                size=self.target_size,
                mode='bicubic',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension
            
            # For complex data m_I, real and imaginary parts need to be handled separately
            real_part = m_I.real
            imag_part = m_I.imag
            
            # Interpolate real and imaginary parts separately
            real_part = torch.nn.functional.interpolate(
                real_part.unsqueeze(0),
                size=self.target_size,
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
            
            imag_part = torch.nn.functional.interpolate(
                imag_part.unsqueeze(0),
                size=self.target_size,
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
            
            # Recombine real and imaginary parts into complex tensor
            m_I = torch.complex(real_part, imag_part)
        
        name = self.npy_filenames[index].split('.')[0]
        return I_alpha, m_I, name, (current_h, current_w) 
##################################################################
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Normalize
import cv2
test_transform = A.Compose([
    Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2()],
    additional_targets={'image45': 'image', 'image90': 'image', 'image135': 'image'},
)
##################################################################################################   
def read_image(path, target_size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    return img
class DataLoader_test_v1(Dataset):
    """
    Test dataset
    
    args:
        npy_dir (str): dataset directory
    """
    def __init__(self, base_dir, target_size=(512, 512)):
        super(DataLoader_test_v1, self).__init__()
        self.base_dir = base_dir
        self.input_0_dir = os.path.join(base_dir, '0')
        self.input_45_dir = os.path.join(base_dir, '45')
        self.input_90_dir = os.path.join(base_dir, '90')
        self.input_135_dir = os.path.join(base_dir, '135')
        
        self.imgnames = fnmatch.filter(os.listdir(self.input_0_dir), '*.png')
        self.tar_size = len(self.imgnames )  # get the size of target
        self.target_size = target_size  # Target size, default to 512x512

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I0 = read_image(os.path.join(self.input_0_dir, self.imgnames[index]), target_size=self.target_size)
        I45 = read_image(os.path.join(self.input_45_dir, self.imgnames[index]), target_size=self.target_size)
        I90 = read_image(os.path.join(self.input_90_dir, self.imgnames[index]), target_size=self.target_size)
        I135 = read_image(os.path.join(self.input_135_dir, self.imgnames[index]), target_size=self.target_size)

        augmented = test_transform(image=I0, image45=I45, image90=I90, image135=I135)
        i0 = augmented['image']
        i45 = augmented['image45']
        i90 = augmented['image90']
        i135 = augmented['image135']
        # Calculate network input
        I_alpha, m_I = self._calc_input(i0, i45, i90, i135)
        current_h, current_w = I_alpha.shape[1], I_alpha.shape[2]

        name = self.imgnames[index].split('.')[0]
        return I_alpha, m_I, name, (current_h, current_w)

    def _calc_input(self, I0, I45, I90, I135):
        """
        Calculate polarization parameters
        args:
            I0, I45, I90, I135: torch tensors with shape (C, H, W)
        returns:
            I_alpha: torch tensor with shape (12, H, W)
        """
        I_alpha = torch.cat([I0, I45, I90, I135], dim=0)  # (12,H,W)
        S0 = (I0 + I45 + I90 + I135) / 2.0  # (C,H,W)
        S1 = I0 - I90  # (C,H,W)
        S2 = I45 - I135  # (C,H,W)
        mI_real = S1 / (S0 + 1e-8)  # (C,H,W)
        mI_imag = S2 / (S0 + 1e-8)  # (C,H,W)
        mI = torch.complex(mI_real, mI_imag)  # (C,H,W)

        return I_alpha, mI
##################################################################################################
def get_training_data(opt, img_options):
    npy_dir = opt.train_dir
    stage = opt.stage
    assert os.path.exists(npy_dir)
    if stage == 'pol':
        return DataLoader_pol(npy_dir, is_train=True, img_options=img_options, target_transform=None)
    elif stage == 'img':
        return DataLoader_img(npy_dir, is_train=True, img_options=img_options, target_transform=None)
    elif stage == 'full':
        return DataLoader_full(npy_dir, is_train=True, img_options=img_options, target_transform=None)
    else:
        raise NotImplementedError


def get_validation_data(opt):
    npy_dir = opt.val_dir
    stage = opt.stage
    assert os.path.exists(npy_dir)
    if stage == 'pol':
        return DataLoader_pol(npy_dir, is_train=False, target_transform=None)
    elif stage == 'img':
        return DataLoader_img(npy_dir, is_train=False, target_transform=None)
    elif stage == 'full':
        return DataLoader_full(npy_dir, is_train=False, target_transform=None)
    elif stage == 'test':
        return DataLoader_test(npy_dir)
    elif stage == 'test_v1':
        return DataLoader_test_v1(npy_dir)
    else:
        raise NotImplementedError
