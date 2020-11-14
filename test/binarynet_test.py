import unittest
import os
import shutil


class TestBinaryNet(unittest.TestCase):

    def test_import(self):
        from ftbnn import binarynet

    def test_import_data(self):
        from ftbnn.binarynet import preprocess_data
        training_data, testing_data = preprocess_data()

    def test_build_model(self):
        from ftbnn.binarynet import build_model
        model = build_model()

    def test_train_test_cifar10(self):
        from ftbnn.binarynet import preprocess_data
        training_data, testing_data = preprocess_data()
        from ftbnn.binarynet import build_model
        model = build_model()
        from ftbnn.binarynet import train_model
        history = train_model((training_data[0][:1000], training_data[1][:1000]), model, epochs=1, batch_size=100)
        from ftbnn.binarynet import test_model
        test_acc = test_model((testing_data[0][:100], testing_data[1][:100]), model)
        print(f"Test accuracy {test_acc * 100:.2f} %")

    def test_save_load_model(self):
        try:
            os.mkdir("./temp/")
        except:
            pass
        from ftbnn.binarynet import build_model
        model = build_model()
        from ftbnn.binarynet import save_model
        save_model(model, path="./temp/")
        from ftbnn.binarynet import load_model
        model = load_model(path="./temp/")
        shutil.rmtree("./temp/")

    def cleanUp(self):
        try:
            shutil.rmtree("./temp")
        except:
            pass

if __name__ == '__main__':
    unittest.main()
