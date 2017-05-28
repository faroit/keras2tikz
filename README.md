# deeptikz
Generate (hopefully) readable tikz code for DNN layer diagrams.

## Why?

Keras has a `summary` and a [builtin graphviz export](https://github.com/fchollet/keras/blob/master/keras/utils/vis_utils.py). Both show the input and output shapes of all layers. However, many papers show the layer parameters, so I've added some common parameters like hidden units (Dense + Recurrent), Kernel size and number of filter (Convolutional), Pooling Dimensions... to the output graph...

## Usage

```python
from keras import applications

model = applications.VGG16(weights=None)
tix_code = gen_tikz_from_model(model)
tex_file = open("model.tex", 'w')
tex_file.write(tex_file)
tex_file.close()
```

![vgg16](https://cloud.githubusercontent.com/assets/72940/26532609/61ff71aa-4405-11e7-9827-6cc4b12550dc.png)
