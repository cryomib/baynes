import os
import unittest
import matplotlib.pyplot as plt

# Import the MatplotlibHelper class from your code
from baynes.plotter import MatplotlibHelper

class MatplotlibHelperTestCase(unittest.TestCase):
    def setUp(self):
        # Create an instance of MatplotlibHelper for testing
        self.helper = MatplotlibHelper(save=True)

    def tearDown(self):
        # Clean up any created figure files
        self.helper.clear_figures()
        self.helper.set_savefig(False)

    def test_new_figure(self):
        # Test creating a new figure
        figure = self.helper.new_figure('plot')
        self.assertIsInstance(figure, plt.Figure)
        self.assertEqual(len(self.helper.figures), 1)

    def test_add_lines(self):
        # Test adding lines to the current figure
        figure = self.helper.new_figure('plot')
        axes = figure.subplots()
        self.helper.add_lines(x_coords=[1], y_coords=[4], label='lines')
        self.assertEqual(len(figure.axes[0].lines), 2)  # Each x/y pair adds 2 lines

    def test_save_figures(self):
        # Test saving the figures
        self.helper.new_figure('plot')
        self.helper.save_figures()
        figure_title = list(self.helper.figures.keys())[0]
        figure_path = f"{self.helper.output_dir}{figure_title}{self.helper.format}"
        self.assertTrue(os.path.exists(figure_path))

if __name__ == '__main__':
    unittest.main()
