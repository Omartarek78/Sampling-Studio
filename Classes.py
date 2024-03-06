import numpy as np
import math
import scipy
import scipy.signal

from scipy import interpolate as interp
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import interp1d

class Signal:
    def _init_(self):
        self.data=None
        self.name=None
        self.frequency=None
        self.sample_duration=None
        self.reconstructed_data=None
        self.time = np.linspace(0, 1, 1000)

    def set_name(self,name):
        self.name=name

    def get_time(self):
         return self.time

    def set_data(self,data):
        self.data=data

    def get_name(self):
        return self.name

    def get_data(self):
        return self.data

    def calc_amplitude(self):
        return max(abs(value) for value in self.data)

    def calc_frequency(self):
        data=self.get_data()
        if(len(data)<2):
            print("there's no enough data")
            return
        sampling_period = data[1][0]-data[0][0]
        sampling_frequency= 1/sampling_period
        self.frequency=sampling_frequency/2
        return self.frequency


    def sampling(self,data, frequency):
        # sampling_period= 1/frequency
        resampled_time = np.linspace(0, 1.998, frequency)
        original_time = self.get_data()[:, 0]
        original_data = data
        f = interp.interp1d(original_time, original_data)
        resampled_data_y = f(resampled_time)
        resampled_data = np.stack((resampled_time, resampled_data_y), axis=1)
        return resampled_data


    # @staticmethod
    # def sinc_interpolation_2(sampled_x, sampled_y, f_sample, new_time_points):
    #     """
    #     Reconstruct a signal using sinc interpolation.

    #     Args:
    #         sampled_x (array-like): The time (or position) values at which the signal is sampled.
    #         sampled_y (array-like): The corresponding sampled signal values.
    #         f_sample (float): The sampling frequency (or sampling rate).
    #         new_time_points (array-like): The time points at which you want to reconstruct the signal.

    #     Returns:
    #         np.array: The reconstructed signal values at new_time_points.
    #     """
    #     time_interval = 1 / f_sample
    #     reconstructed_signal = np.zeros(len(new_time_points))

    #     for n, sample in enumerate(sampled_y):
    #         reconstructed_signal += sample * np.sinc((new_time_points - sampled_x[n]) / time_interval)

    #     return reconstructed_signal
    @staticmethod
    def o_reconstruct(t_values,f_sample,time_sampled,y_sampled):
        y_recon = np.zeros_like(t_values)
        for point in range(len(t_values)):
            y_recon[point] = np.sum(y_sampled * np.sinc((t_values[point] - time_sampled) * f_sample))
        return y_recon

    
    def difference(self,og_data,reconstructed_data):
        return (og_data-reconstructed_data)
