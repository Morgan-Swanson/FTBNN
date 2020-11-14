import unittest

class TestBinaryNet(unittest.TestCase):

    def test_import(self):
        from ftbnn import binarynet

    def test_import_data(self):
        from ftbnn.binarynet import preprocess_data
        training_data, testing_data = preprocess_data()

    def test_build_model(self):
        from ftbnn.binarynet import build_model
        model = build_model()

    def test_train_cifar10(self):
        from ftbnn.binarynet import preprocess_data
        training_data, testing_data = preprocess_data()
        from ftbnn.binarynet import build_model
        model = build_model()
        from ftbnn.binarynet import train_model
        history = train_model(training_data, model, epochs=1)

if __name__ == '__main__':
    unittest.main()
