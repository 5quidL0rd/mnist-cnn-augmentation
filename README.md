# mnist-cnn-augmentation
A PyTorch implementation of a Convolutional Neural Network (CNN) for classifying the MNIST dataset, featuring data augmentation and real-time training visualizations.

Download the code and run it in the command line to download the MNIST dataset and train your neural networks. 
---

## **Prerequisites**

At the beginning of this project I would advise creating a virtual environment for the project to run in: 

```bash
python3 -m venv dl_env  # Create a virtual environment
source dl_env/bin/activate  # Activate it (Linux/macOS)
# On Windows, use: dl_env\Scripts\activate
```
You should see (dl_env) to the left of your shell prompt now. 

Additionally, ensure you have the following installed:

- **Python 3.x**
- **PyTorch**
- **Matplotlib**

To install the necessary dependencies, run the following command:

```bash
pip install torch torchvision torchaudio numpy pandas matplotlib scikit-learn

```

## **Feedforward Neural Network for the first test**

Run with the following command:

```bash
python3 mnist_pytorch.py
```

This will: 
1) Download the MNIST dataset
2) Train a simple 2-layer neural network
3) Print loss after each epoch


## **CNN Neural Network**

CNN are a much more effective at image classification as well shall see.

```bash
python3 mnist_cnn.py
```

After you run the second file in this Github as shown above you will see that this CNN is much more effective at minimizing our loss function. 
To summarize, this cnn code is more effective as it utilizes:
1) Local feature detection: CNNS use convolutional layers that apply filter sto small regions of the input, so they can detect local patterns more effecitvely
2) Weight Sharing: This CNN model uses the same filter (same weights) across the entire image so it reduces overall parameter number.
3) Spatial Hierarchy: CNNs preserve spatial structure of images, so they can recognize more complex patterns our Feedforward wouldn't
4) Pooling Layers: CNNS include pooling layers that reduce the spatial size of the input.
5) Also, we use data augmentation in the cnn code which artificially increases the size of the data by roatating the images, scaling images, etc.

## **CNN Neural Network with Visuals**

Thus far, we only can see training epochs and percentages outputted to our command line. Some pictures would make things more interesting and informative to boot. 
Run the mnist_cnn_visuals.py code in the github. 

```bash
python3 mnist_cnn_visuals.py
```

Now, after you finish running your cnn take a look at what files are in your directory.

```bash
ls
```

You should see different images you can open up and take a look at. I am using xdg for this but there are different ways to open it up.

```bash
xdg-open sample_predictions_epoch_2.png
```

You can now observe the MNIST dataset and see how your model does on classification. 




## **Further Research** 

And that's it! 

If you would like to continue to research this topic:

Pytorch documentation: https://pytorch.org/docs/stable/index.html

Paper on Document Recognition: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

3Blue1Brown CNN video: https://www.youtube.com/watch?v=KuXjwB4LzSA

