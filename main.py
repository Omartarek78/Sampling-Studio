from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import os
import numpy as np
from pyqtgraph import ScatterPlotItem
from scipy.interpolate import interp1d

import Classes
import pyqtgraph as pg
# import scipy.signal

from os import path
import sys
# import UI file
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "style.ui"))


# initiate UI file
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.handle_buttons()
        self.intialize()

    def intialize(self):
        self.samplingFrequencySlider.setMinimum(1)
        self.samplingFrequencySlider.setMaximum(99)
        # self.samplingFrequencySlider.setMinimum(1)
        # self.samplingFrequencySlider.setMaximum(99)
        self.samplingFrequencySlider.setSingleStep(1)
        self.samplingFrequencySlider.valueChanged.connect(self.freqSliderChanged)

        self.snrSlider.setMinimum(0)
        self.snrSlider.valueChanged.connect(self.snrSliderChanged)
        

        self.fmax = None

        pg.setConfigOptions(antialias=True)

        self.components = []  # List to store sinusoidal components
        self.composite_signal = None  # To store the composite signal
        self.componentsComboBox.clear()
        self.selected_component = None
        self.componentsComboBox.currentIndexChanged.connect(self.selected_component_changed)

        self.sampled_signal = None  # To store the sampled signal
        self.sampled_indices = None
        self.sampling_frequency = None
        self.sampling_frequency_load = None

        self.rec_y = None
        self.y_reconstructed = None

        self.noise = None  # To store the generated noise
        self.composite_signal_with_noise = None
        self.load_signal_with_noise = None

        self.error_threshold = 0

        # self.originalgraphicsView.setYRange(-10, 10)
        # self.differencegraphicsView.setYRange(-10, 10)
        self.reconstructedgraphicsView.setYRange(-10, 10)

    def handle_buttons(self):
        self.addSignalButton.clicked.connect(self.add_signal_component)
        self.removeComponentsButton.clicked.connect(self.remove_selected_component)
        self.browseFilesButton.clicked.connect(self.load_data)
        self.deleteSignalButton.clicked.connect(self.deleteSignal)

    def load_data(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Data Files (*.dat *.csv)")
        if filepath:
            _, extension = os.path.splitext(filepath)

            if not path.exists(filepath):
                QMessageBox.critical(self, "File Not Found", f"Could not find file at {filepath}.")
                return

            data = None

            if extension == '.dat':
                # Read the .dat file as 16-bit integers
                data = np.fromfile(filepath, dtype=np.int16)
            elif extension == '.csv':
                data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            self.orginalSignal = Classes.Signal()
            self.samplingFrequencySlider.setMaximum(1000)
            self.orginalSignal.set_data(data[:1000, :1000])
            self.orginalSignal.set_name("originalSignal")
            self.plotData()

    def plotData(self):
        self.originalgraphicsView.clear()
        self.samplingFrequencySlider.setMaximum(99)
        self.originalgraphicsView.addItem(pg.PlotDataItem(self.orginalSignal.get_data()))
        self.reconstructedgraphicsView.setXRange(0, len(self.orginalSignal.get_data()[0]))
        self.originalgraphicsView.setXRange(0, len(self.orginalSignal.get_data()[0]))
        self.differencegraphicsView.setXRange(0, len(self.orginalSignal.get_data()[0]))
        self.differencegraphicsView.setYRange(-1, 1)

    def add_signal_component(self):
        self.originalgraphicsView.clear()
        frequency = self.signalFrequencySlider.value()
        magnitude = self.signalAmplitudeSlider.value()
        self.add_component(frequency, magnitude)
        self.composite_signal = self.generate_composed_signal()

        self.samplingFrequencySlider.setValue(1)
        self.snrSlider.setValue(0)

    def add_component(self, frequency, magnitude):
        """Add a sinusoidal component to the composite signal."""
        self.components.append((frequency, magnitude))
        # Get the component count, which will be used as the signal number
        signal_number = len(self.components) - 1

        # Create a string to represent the component
        component_text = f'Signal {signal_number} - Freq: {frequency}, Amp: {magnitude}'

        # Add the component to the ComboBox
        self.componentsComboBox.addItem(component_text)

    def generate_composed_signal(self, duration=1, sample_rate=1000):
        """Generate the composite signal based on added components."""

        self.originalgraphicsView.clear()

        # Create an array to store the composite signal
        composite_signal = np.zeros(int(duration * sample_rate))

        # Generate time values
        # t = np.linspace(0, duration, len(composite_signal), endpoint=False)
        t = np.arange(0, duration, 1/len(composite_signal))

        # Iterate through components and add their contributions
        for frequency, magnitude in self.components:
            component_signal = magnitude * np.sin(2*np.pi*frequency * t)
            composite_signal += component_signal

        self.composite_signal = composite_signal
        self.plot_signal_on_original_view(t, self.composite_signal)
        return self.composite_signal

    def plot_signal_on_original_view(self, t, signal):
        # Create a PlotDataItem for the signal
        data_item = pg.PlotDataItem(t, signal)
        self.originalgraphicsView.addItem(data_item)

    def find_max_frequency(self, tuple_list):
        max_frequency = None

        for frequency, _ in tuple_list:
            if max_frequency is None or frequency > max_frequency:
                max_frequency = frequency

        return max_frequency

    def freqSliderChanged(self):
        if not self.components:
            self.update_sampling_load()

        else:
            self.update_sampling()

    def update_sampling(self):
        if self.composite_signal_with_noise is not None:
            composite_signal = self.composite_signal_with_noise
        elif self.composite_signal is not None:
            composite_signal = self.composite_signal
        else:
            return  # No signal to sample

        self.samplingFrequencySlider.setMaximum(600)
        
        self.sampling_frequency = self.samplingFrequencySlider.value()
        t = np.linspace(0, 1, len(composite_signal), endpoint=False)

        # Interpolate the signal to create a more densely sampled version
        # interp_func = interp1d(t, self.composite_signal, kind='linear')  # self.composite_signal or composite_signal????????
        interp_func = interp1d(t, composite_signal, kind='linear')  # self.composite_signal or composite_signal????????
        time_sampled = np.arange(0, 1, 1/self.sampling_frequency)
        sampled_signal = interp_func(time_sampled)
        self.sampled_signal = sampled_signal

        # sampled_scatter = ScatterPlotItem()
        # sampled_scatter.setData(time_sampled, sampled_signal, symbol='o', brush=(255, 0, 0), size=4)
        # self.originalgraphicsView.addItem(sampled_scatter)

        self.originalgraphicsView.clear()
        # Plot the composite signal (with or without noise)
        pen = pg.mkPen(color='gray', width=2)
        self.originalgraphicsView.plot(t, composite_signal, pen=pen)
        # Plot the sampled points on the composite signal
        pen = pg.mkPen(color='b', symbol='o', symbolBrush=(0, 0, 255), symbolSize=6)  # Blue points
        self.originalgraphicsView.plot(time_sampled, sampled_signal, pen=None, symbol='o', size=3)

        self.recover_signal()
        self.fmax = self.find_max_frequency(self.components)
        self.actualFrequencyLCD.display(self.sampling_frequency)
        normalized_frequency = self.sampling_frequency / self.fmax
        self.normalizedFrequencyLCD.display(normalized_frequency)

    def update_sampling_load(self):
        if self.load_signal_with_noise is not None:
            load_signal = self.load_signal_with_noise
        elif self.orginalSignal is not None:
             load_signal = self.orginalSignal.get_data()[:, 1]
        else:
             return  # No signal to sample

        self.samplingFrequencySlider.setMaximum(1000)
        self.sampling_frequency_load = self.samplingFrequencySlider.value()
        num_samples = len(load_signal) // self.sampling_frequency_load
        self.sampled_indices_load = list(range(0, len(load_signal), num_samples))
        self.sampled_signal_load = self.orginalSignal.sampling(load_signal,
                                                               self.sampling_frequency_load)
        # Clear the existing data items in the originalgraphicsView
        self.originalgraphicsView.clear()
        # Plot the composite signal (with or without noise)
        pen = pg.mkPen(color='gray', width=2)
        t_load = np.linspace(0, 2, len(load_signal), endpoint=False)
        self.originalgraphicsView.plot(t_load, load_signal, pen=pen)
        # Plot the sampled points on the composite signal
        self.originalgraphicsView.plot(self.sampled_signal_load, pen=None, symbol='o', size=2)

        self.reconstruct_load(self.orginalSignal.calc_frequency())

        fmax = self.orginalSignal.calc_frequency()

        self.actualFrequencyLCD.display(self.sampling_frequency_load)
        normalized_frequency = self.sampling_frequency_load / fmax

        self.normalizedFrequencyLCD.display(normalized_frequency)

    def reconstruct_load(self, freq):
        self.reconstructedgraphicsView.clear()
        # Get the original signal data
        data = self.orginalSignal.get_data()

        # Extract the time values and the signal values
        t_values = data[:, 0]
        original_signal = data[:, 1]

        # Resample the signal
        sampling_rate = self.samplingFrequencySlider.value()
        resampled_points = self.orginalSignal.sampling(original_signal, sampling_rate)

        # Extract the y-values of the resampled points
        resampled_points_y = resampled_points[:, 1]
        resampled_points_x = resampled_points[:, 0]

        # y_reconstructed = Classes.Signal.o_reconstruct(t_values, sampling_rate, resampled_points_x, resampled_points_y)
        y_reconstructed = Classes.Signal.o_reconstruct(t_values, freq, resampled_points_x, resampled_points_y)
        self.reconstructedgraphicsView.clear()

        self.rec_y = y_reconstructed
        self.reconstructedgraphicsView.addItem(pg.PlotDataItem(t_values, y_reconstructed))
        self.error(t_values)

    def recover_signal(self):
        if self.sampled_signal is not None:
            # Create a time vector for the recovered signal
            t_recovered = np.linspace(0, 1, len(self.composite_signal), endpoint=False)

            t_sampled = np.linspace(0, 1, len(self.sampled_signal),
                                    endpoint=False)  # adjusted the length to match self.sampled_indices

            self.recs = Classes.Signal.o_reconstruct(t_recovered, self.sampling_frequency, t_sampled, self.sampled_signal)

            self.reconstructedgraphicsView.clear()

            data_item = pg.PlotDataItem(t_recovered, self.recs)
            self.reconstructedgraphicsView.addItem(data_item)
            self.update_difference_signal()

    def error(self, time):
        self.differencegraphicsView.clear()
        if self.load_signal_with_noise is not None:
            error = self.orginalSignal.difference(self.load_signal_with_noise, self.rec_y)
        else:
            error = self.orginalSignal.difference(self.orginalSignal.get_data()[:, 1], self.rec_y)
        error_above_threshold = [err if err > self.error_threshold else 0 for err in error]

        self.differencegraphicsView.addItem(pg.PlotDataItem(time, error_above_threshold))

    def calculate_difference_signal(self):
        # if self.composite_signal_with_noise is not None:
        #     difference_signal = self.composite_signal_with_noise - self.recs
        # else:
        #     difference_signal = self.composite_signal - self.recs
        #
        # return difference_signal

        difference_signal = self.composite_signal - self.recs

        return difference_signal


    def update_difference_signal(self):
        difference_signal = self.calculate_difference_signal()
        if difference_signal is not None:
            self.differencegraphicsView.clear()
            t = np.linspace(0, 1, len(self.composite_signal), endpoint=False)
            data_item = pg.PlotDataItem(t, difference_signal)
            self.differencegraphicsView.addItem(data_item)
            self.differencegraphicsView.setYRange(-10, 10)

    def snrSliderChanged(self):
        if not self.components:
            self.update_noise_load()
        else:
            self.update_noise()

    def update_noise_load(self):
        snr_slider_value = self.snrSlider.value()
        # self.samplingFrequencySlider.setValue(1)
        # Invert the slider value to make 0 correspond to maximum noise and 100 correspond to no noise
        inverted_snr = 100 - snr_slider_value

        if self.orginalSignal.get_data() is not None:
            # Generate noise based on the inverted SNR level
            self.noise = self.generate_noise(inverted_snr, self.orginalSignal.get_data()[:, 1])

            # Add the noise to the composite signal
            self.load_signal_with_noise = self.orginalSignal.get_data()[:, 1] + self.noise

            # Clear any existing data items in the originalgraphicsView
            self.originalgraphicsView.clear()
            self.update_sampling_load()

            # Plot the composite signal with noise
            t = np.linspace(0, 2, len(self.load_signal_with_noise), endpoint=False)
            pen = pg.mkPen(color='gray', width=2)
            self.originalgraphicsView.plot(t, self.load_signal_with_noise, pen=pen)
            self.snrLCD.display(snr_slider_value)

    def update_noise(self):
        snr_slider_value = self.snrSlider.value()
        # self.samplingFrequencySlider.setValue(1)
        # Invert the slider value to make 0 correspond to maximum noise and 100 correspond to no noise
        inverted_snr = 100 - snr_slider_value


        if self.composite_signal is not None:
            # Generate noise based on the inverted SNR level
            self.noise = self.generate_noise(inverted_snr, self.composite_signal)

            # Add the noise to the composite signal
            self.composite_signal_with_noise = self.composite_signal + self.noise

            # Clear any existing data items in the originalgraphicsView
            self.originalgraphicsView.clear()

            # Plot the composite signal with noise
            t = np.linspace(0, 1, len(self.composite_signal_with_noise), endpoint=False)
            pen = pg.mkPen(color='gray', width=2)
            self.originalgraphicsView.plot(t, self.composite_signal_with_noise, pen=pen)
            self.update_sampling()#ssssssss
            self.snrLCD.display(snr_slider_value)

    def generate_noise(self, snr_level, signal):
        """Generate noise based on the SNR level and signal magnitude."""
        signal_power = np.var(signal)
        noise_power = signal_power / (10 ** (snr_level / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return noise

    def deleteSignal(self):
        self.originalgraphicsView.clear()
        self.reconstructedgraphicsView.clear()
        self.differencegraphicsView.clear()
        self.components = []
        self.componentsComboBox.clear()

    def selected_component_changed(self, index):
        if 0 <= index < len(self.components):
            # Get the selected component's frequency and amplitude from the components list
            frequency, amplitude = self.components[index]
            self.selected_component = (frequency, amplitude)

        else:
            print("Selected index is out of range or components list is empty.")

    def remove_selected_component(self):
        if hasattr(self, 'selected_component'):
            # Remove the selected component by frequency and amplitude
            frequency, amplitude = self.selected_component
            self.remove_component(frequency, amplitude)
            # Remove the item from the combo box
            self.componentsComboBox.removeItem(self.componentsComboBox.currentIndex())
            self.samplingFrequencySlider.setValue(1)
            self.snrSlider.setValue(0)

    def remove_component(self, frequency, amplitude):
        """Remove a sinusoidal component with the specified frequency and amplitude."""
        # Create a list to store the new components without the specified frequency and amplitude
        new_components = [(f, a) for f, a in self.components if (f, a) != (frequency, amplitude)]

        # Update the self.components list
        self.components = new_components

        # Update the self.composite_signal
        self.composite_signal = self.generate_composed_signal()  # Regenerate the composite signal
        print(self.components)

   

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    # app.setStyleSheet(qdarkstyle.load_stylesheet())
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
