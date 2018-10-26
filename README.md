# Norm Matters
**See https://github.com/eladhoffer/convNet.pytorch for updated version of this code**

This code was was used to implement [Norm matters: efficient and accurate normalization schemes in deep networks - Hoffer, Banner, Golan, Soudry (2018): ](https://arxiv.org/abs/1803.01814)
```
@inproceedings{hoffer2018norm,
  title={Norm matters: efficient and accurate normalization schemes in deep networks},
  author={Hoffer, Elad and Banner, Ron and Golan, Itay and Soudry, Daniel},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
```

It is based off [imagenet example in pytorch](https://github.com/pytorch/examples/tree/master/imagenet) with some helpful additions such as:
  - Training on several datasets other than imagenet
  - Complete logging of trained experiment
  - Graph visualization of the training/validation loss and accuracy
  - Definition of preprocessing and optimization regime for each model

## Dependencies

- [pytorch](<http://www.pytorch.org>)
- [torchvision](<https://github.com/pytorch/vision>) to load the datasets, perform image transforms
- [pandas](<http://pandas.pydata.org/>) for logging to csv
- [bokeh](<http://bokeh.pydata.org>) for training visualization


## Data
- Configure your dataset path at **data.py**.
- To get the ILSVRC data, you should register on their site for access: <http://www.image-net.org/>


## Model configuration

Network model is defined by writing a <modelname>.py file in <code>models</code> folder, and selecting it using the <code>model</code> flag. Model function must be registered in <code>models/\_\_init\_\_.py</code>
The model function must return a trainable network. It can also specify additional training options such optimization regime (either a dictionary or a function), and input transform modifications.

e.g for a model definition:

```python
class Model(nn.Module):

    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.model = nn.Sequential(...)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            {'epoch': 15, 'lr': 1e-3, 'weight_decay': 0}
        ]

        self.input_transform = {
            'train': transforms.Compose([...]),
            'eval': transforms.Compose([...])
        }
    def forward(self, inputs):
        return self.model(inputs)
        
 def model(**kwargs):
        return Model()
```
