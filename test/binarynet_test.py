import unittest

class TestBinaryNet(unittest.TestCase):

    def test_import(self):
        from ftbnn import binarynet

    def test_import_data(self):
        from ftbnn import binarynet
        binarynet.preprocess_data()

    def test_build_model(self):
        from ftbnn import binarynet
        binarynet.build_model()

    def test_train_cifar10(self):
        from ftbnn.binarynet import plot_results, train_model, preprocess_data, build_model
        plot_results(train_model(preprocess_data(), build_model()))

if __name__ == '__main__':
    unittest.main()