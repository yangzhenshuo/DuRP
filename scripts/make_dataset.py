import numpy as np
import argparse
import polanalyser as pa
import os, random
from tqdm import tqdm
import string


class Synthesizer:
    def __init__(self, size, sv_A_inf, sv_beta):
        self.H, self.W, self.C = size
        self.sv_A_inf = sv_A_inf
        self.sv_beta = sv_beta
    def _generate_random_params(self, factor, eps=0.05):
        #### A_inf ####
        A_inf_base = np.random.uniform(0.85, 0.95)
        A_inf_fluctuation = A_inf_base * eps
        if self.sv_A_inf:
            A_inf = np.sort(
                np.float32(
                    np.random.uniform(-A_inf_fluctuation, A_inf_fluctuation,
                                      (self.H, self.W, self.C)) + A_inf_base
                )
            ) # (H, W, C)  A_inf_r < A_inf_g < A_inf_b if RGB
        else:
            A_inf = np.sort(
                np.float32(
                    np.random.uniform(-A_inf_fluctuation, A_inf_fluctuation,
                                      (self.C,)) + A_inf_base
                )
            ) # (C,)  A_inf_r < A_inf_g < A_inf_b if RGB
        #### beta ####
        start_value = 0.0025 + 0.002 * factor
        end_value = 0.0035 + 0.002 * factor
        beta_base = np.random.uniform(start_value, end_value)
        beta_fluctuation = beta_base * eps
        if self.sv_beta:
            beta = np.sort(
                np.float32(
                    np.random.uniform(-beta_fluctuation, beta_fluctuation,
                                      (self.H, self.W, self.C)) + beta_base
                )
            ) # (H, W, C)  beta_r < beta_g < beta_b if RGB
        else:
            beta = np.sort(
                np.float32(
                    np.random.uniform(-beta_fluctuation, beta_fluctuation,
                                      (self.C,)) + beta_base
                )
            ) # (C,)  beta_r < beta_g < beta_b if RGB
        #### P_A ####
        P_A_base = np.random.uniform(0.05, 0.4)
        P_A_fluctuation = P_A_base * eps
        P_A = np.clip(
            -np.sort(
                -np.float32(
                    np.random.uniform(-P_A_fluctuation, P_A_fluctuation,
                                      (self.H, self.W, self.C)) + P_A_base
                )
            ), a_min=0, a_max=1
        ) # (H, W, C) P_A_r > P_A_g > P_A_b if RGB
        #### theta_A ####
        theta_A_mean = np.random.uniform(-np.pi/4, np.pi/4)
        theta_A_std = np.abs(theta_A_mean) * eps
        theta_A = np.clip(
            np.float32(
                np.random.normal(theta_A_mean, theta_A_std,
                                 (self.H, self.W, self.C))
            ), a_min=-np.pi/4, a_max=np.pi/4
        )
        theta_A = np.mod(theta_A, np.pi)/np.pi # range-->[0, 1]
        m_A = P_A * np.exp(1j * 2 *np.pi* theta_A)
        return {'A_inf': A_inf, 'beta': beta, 'm_A': m_A}
    def _generate_transmission(self, params):
        try:
            beta = params.pop('beta')
            depth = params.pop('depth')
        except KeyError:
            raise ValueError('beta or depth not found in params')
        if len(depth.shape) < 3:
            depth = depth.reshape((self.H, self.W, 1))
        depth = (depth - depth.min()) / (depth.max() - depth.min())* 255.0
        # depth = depth.astype(np.uint8)
        transmission = np.clip(np.exp(-beta * depth, dtype=np.float32),a_min=0.1, a_max=1)
        return {'transmission': transmission}
    def _generate_A(self, params):
        try:
            A_inf = params['A_inf']
            m_A = params['m_A']
            transmission = params['transmission']
        except KeyError:
            raise ValueError('A_inf, P_A, theta_A or transmission not found in params')
        P_A = np.abs(m_A)
        theta_A = np.mod(np.angle(m_A)/2, np.pi)/np.pi # range-->[0, 1]
        A_0 = (A_inf * (1 - transmission) * (1 + P_A * np.cos(2 * np.pi * theta_A)) / 2).astype(np.float32)  # (H, W, C)
        A_45 = (A_inf * (1 - transmission) * (1 + P_A * np.sin(2 * np.pi * theta_A)) / 2).astype(np.float32)  # (H, W, C)
        A_90 = (A_inf * (1 - transmission) * (1 - P_A * np.cos(2 * np.pi * theta_A)) / 2).astype(np.float32)
        A_135 = (A_inf * (1 - transmission) * (1 - P_A * np.sin(2 * np.pi * theta_A)) / 2).astype(np.float32)
        A_list = [A_0, A_45, A_90, A_135]
        return {'A_list': A_list}
    def _generate_D(self, params):
        try:
            angles = params['angles']
            full_pol = params.pop('full_pol')
            transmission = params.pop('transmission')
        except KeyError:
            raise ValueError('angles or full_pol or transmission not found in params')
        R_0 = full_pol[:,:,:,0]
        R_45 = full_pol[:,:,:,1]
        R_90 = full_pol[:,:,:,2]
        R_135 = full_pol[:,:,:,3]
        R = (R_0 + R_45 + R_90 + R_135)/4
        D_0 = R_0 * transmission
        D_45 = R_45 * transmission
        D_90 = R_90 * transmission
        D_135 = R_135 * transmission
        #### D_polarization ####
        D_list = [D_0, D_45, D_90, D_135]
        D_stokes = pa.calcStokes(D_list, angles)
        P_D = np.clip(np.float32(pa.cvtStokesToDoLP(D_stokes)), a_min=0, a_max=1)
        theta_D = np.float32(pa.cvtStokesToAoLP(D_stokes)/np.pi) # range-->[0, 1]
        m_D = P_D * np.exp(1j * 2 * np.pi * theta_D)
        return {'D_list': D_list, 'm_D': m_D, 'R': R}
    def _generate_I(self, params):
        try:
            angles = params['angles']
            A_list = params.pop('A_list')
            D_list = params.pop('D_list')
        except KeyError:
            raise ValueError('A or D not found in params')
        I_0 = A_list[0] + D_list[0]
        I_45 = A_list[1] + D_list[1]
        I_90 = A_list[2] + D_list[2]
        I_135 = A_list[3] + D_list[3]
        D = np.mean(D_list, axis=0)
        I = (I_0 + I_45 + I_90 + I_135)/4
        K = np.clip(D / (I + 1e-8), a_min=0, a_max=1)
        #### add noise ####
        I_0 = np.float32(np.clip(np.random.normal(I_0, I_0*0.005), a_min=0, a_max=1))
        I_45 = np.float32(np.clip(np.random.normal(I_45, I_45*0.005), a_min=0, a_max=1))
        I_90 = np.float32(np.clip(np.random.normal(I_90, I_90*0.005), a_min=0, a_max=1))
        I_135 = np.float32(np.clip(np.random.normal(I_135, I_135*0.005), a_min=0, a_max=1))
        I_alpha = np.concatenate([I_0, I_45, I_90, I_135],  axis=-1) # (H, W, 4*C)
        I_demosaiced_list = [I_0, I_45, I_90, I_135]
        I_stokes = pa.calcStokes(I_demosaiced_list, angles)
        P_I = np.clip(np.float32(pa.cvtStokesToDoLP(I_stokes)), a_min=0, a_max=1)
        theta_I = np.float32(pa.cvtStokesToAoLP(I_stokes)/np.pi) # range-->[0, 1]
        m_I = P_I * np.exp(1j * 2 * np.pi * theta_I)
        return {'I_alpha': I_alpha, 'm_I': m_I, 'K': K}

class DatasetMaker(Synthesizer):
    def __init__(self, size, sv_A_inf, sv_beta, input_dir, output_dir, enlarge_factor):
        super(DatasetMaker, self).__init__(size, sv_A_inf, sv_beta)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.enlarge_factor = enlarge_factor
        self.ouput_input_names = ['I_alpha', 'm_I']
        self.ouput_target_names = ['R', 'm_A', 'A_inf', 'm_D', 'K']
    def _generate_data(self, full_pol, depth, factor):
        #### generate random params ####
        params = self._generate_random_params(factor)
        #### add full_pol ,depth and angles ####
        angles = np.deg2rad([0, 45, 90, 135])
        params.update({'full_pol': full_pol, 'depth': depth, 'angles': angles})
        #### generate transmission ####
        params_transmission = self._generate_transmission(params)
        params.update(params_transmission)
        #### generate A ####
        params_A = self._generate_A(params)
        params.update(params_A)
        #### generate D ####
        params_D = self._generate_D(params)
        params.update(params_D)
        #### generate I ####
        params_I = self._generate_I(params)
        params.update(params_I)
        return params
    def make(self):
        full_pol_path = os.path.join(self.input_dir, 'full_pol')
        depth_path = os.path.join(self.input_dir, 'depth')
        file_names = os.listdir(full_pol_path)
        full_pol_files = [file for file in file_names if file.lower().endswith('.npy')]
        for idx, full_pol_file in enumerate(tqdm(full_pol_files), 0):
            name = full_pol_file.split('.')[0]
            full_pol = np.load(os.path.join(full_pol_path, full_pol_file))
            depth = np.load(os.path.join(depth_path, full_pol_file))
            for i in range(self.enlarge_factor):
                factor = i
                params = self._generate_data(full_pol, depth, factor)
                prefix = name + string.ascii_lowercase[i]
                for ouput_name in self.ouput_input_names:
                    output_dir = os.path.join(self.output_dir, 'input', ouput_name)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    try:
                        value = params.get(ouput_name)
                    except KeyError:
                        raise ValueError('ouput_name not found in params')
                    if np.isnan(value).any() or np.isinf(value).any():
                        print(f'nan or inf value in {ouput_name} it name is {prefix}')
                    if len(value.shape) != 3:
                        print(f'{ouput_name} shape is not 3 it name is {prefix}')
                    np.save(os.path.join(output_dir, prefix+'.npy'), value)
                for ouput_name in self.ouput_target_names:
                    output_dir = os.path.join(self.output_dir, 'target', ouput_name)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    try:
                        value = params.get(ouput_name)
                    except KeyError:
                        raise ValueError('ouput_name not found in params')
                    if np.isnan(value).any() or np.isinf(value).any():
                        print(f'nan or inf value in {ouput_name} it name is {prefix}')
                    if len(value.shape) != 3:
                        print(f'{ouput_name} shape is not 3 it name is {prefix}')
                    np.save(os.path.join(output_dir, prefix+'.npy'), value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate dataset')
    parser.add_argument('--size', nargs='*', default=(512, 512, 3), type=int, help='image size')
    parser.add_argument('--sv_A_inf', default=1, type=int, help='spatially variant A_infinity')
    parser.add_argument('--sv_beta', default=1, type=int, help='spatially variant beta')
    parser.add_argument('--input_dir', required=False, default='raw_data', type=str, help='base dir for original data')
    parser.add_argument('--output_dir', required=False, default='datas', type=str, help='output dir')
    parser.add_argument('--mode', required=False, default='train', type=str, choices=['train', 'test'], help='mode')
    parser.add_argument('--enlarge_factor', default=3, type=int, help='enlarge data factor')
    args = parser.parse_args()
    
    base_dir = os.getcwd()
    input_dir = os.path.join(base_dir, args.input_dir, args.mode)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    output_dir = os.path.join(base_dir, args.output_dir, args.mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # set random seed
    random.seed(3407)
    np.random.seed(3407)
    dataset_maker = DatasetMaker(size=args.size,
                                 sv_A_inf=args.sv_A_inf,
                                 sv_beta=args.sv_beta,
                                 input_dir=input_dir,
                                 output_dir=output_dir,
                                 enlarge_factor=args.enlarge_factor)
    dataset_maker.make()
    print('done')
    
    
    