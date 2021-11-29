import numpy as np


class FullyConnected:
    def __init__(self, num_beams, num_ant, batch_size, mode='orig', accum=False):
        self.num_beams = num_beams
        self.num_ant = num_ant
        self.batch_size = batch_size
        self.count = 0
        self.thetas = self.thetaInit()
        self.W = np.zeros([self.num_beams, self.num_ant])
        self.A = np.zeros([2 * self.num_beams, 1])
        self.mode = mode
        self.accum = accum
        self.state = []
        self.grad = np.zeros([self.num_beams, self.num_ant])
        # Adam
        self.m_t = np.zeros([self.num_beams, self.num_ant])
        self.v_t = np.zeros([self.num_beams, self.num_ant])
        self.time_step = 0

    def thetaInit(self):
        init_thetas = 2 * np.pi * np.random.rand(self.num_beams, self.num_ant)
        return init_thetas

    def forward(self, ch, val_mode=False):  # ch: (2 * num_ant, 1), organized as (h_r, h_i)
        if val_mode:
            w_r = (1 / np.sqrt(self.num_ant)) * np.cos(self.thetas)
            w_i = (1 / np.sqrt(self.num_ant)) * np.sin(self.thetas)
            W_block = np.block([[w_r, w_i], [-w_i, w_r]])
            if ch.shape[0] == W_block.shape[1]:
                A = np.matmul(W_block, ch)
                return A
            else:
                ValueError('Error: dimensions mismatch! Please check FullyConnected.forward, validation mode!')
        else:
            if self.mode == 'orig':
                self.state = ch
            elif self.mode == 'recon':
                pass
            else:
                ValueError('Set mode properly.')
            w_r = (1 / np.sqrt(self.num_ant)) * np.cos(self.thetas)
            w_i = (1 / np.sqrt(self.num_ant)) * np.sin(self.thetas)
            self.W = w_r + 1j * w_i
            W_block = np.block([[w_r, w_i], [-w_i, w_r]])
            if ch.shape[0] == W_block.shape[1]:
                self.A = np.matmul(W_block, ch)  # self.A.shape: (2 * num_beams,) organized as (a_r, a_i)
            else:
                ValueError('Error: dimensions mismatch! Please check FullyConnected.forward!')
            return self.A

    def backward(self, dydx):
        h = []
        if self.mode == 'orig':
            h = self.state
        elif self.mode == 'recon':
            h = self.estimate()
        else:
            ValueError('Please set mode properly!')
        h_r = h[:self.num_ant]
        h_i = h[self.num_ant:]
        dxdz = np.zeros([2 * self.num_beams, self.num_ant])
        for ii in range(self.num_beams):
            theta_ii = self.thetas[ii, :]
            dxdz[ii, :] = (1 / 8) * (-h_r * np.sin(theta_ii) + h_i * np.cos(theta_ii))
            dxdz[ii + self.num_beams, :] = (1 / 8) * (-h_i * np.sin(theta_ii) - h_r * np.cos(theta_ii))
        if self.accum:
            if self.count < self.batch_size:
                self.grad = self.grad + np.matmul(dydx, dxdz)
                self.count += 1
            else:
                self.count = 0
                self.grad = np.zeros([self.num_beams, self.num_ant])
                self.grad = self.grad + np.matmul(dydx, dxdz)
                self.count += 1
        else:
            self.grad = np.matmul(dydx, dxdz)
        return self.grad

    def estimate(self):
        A_complex = self.A[:self.num_beams] + 1j * self.A[self.num_beams:]
        W_conj = np.conj(self.W)
        h_est = np.matmul(np.linalg.pinv(W_conj, rcond=1e-3), A_complex)
        h_est = np.concatenate((np.real(h_est), np.imag(h_est)), axis=0)
        return h_est

    def update(self, lr=0.1, beta_1=0.9, beta_2=0.999):
        self.time_step = self.time_step + 1
        self.m_t = beta_1 * self.m_t + (1 - beta_1) * self.grad
        self.v_t = beta_2 * self.v_t + (1 - beta_2) * np.power(self.grad, 2)
        m_t_corr = (1 / (1 - np.power(beta_1, self.time_step))) * self.m_t
        v_t_corr = (1 / (1 - np.power(beta_2, self.time_step))) * self.v_t
        self.thetas = self.thetas - lr * np.power(v_t_corr, -0.5) * m_t_corr
        return self.thetas
