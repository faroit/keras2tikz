from keras import applications
import dot2tex as d2t
import pydot_ng as pydot
import keras.layers.convolutional
import keras.layers.recurrent


def model_to_dot(model,
                 rankdir='TB'):
    """Convert a Keras model to dot format.
    # Arguments
        model: A Keras model instance.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
    # Returns
        A `pydot.Dot` instance representing the Keras model.
    """
    from keras.layers.wrappers import Wrapper
    from keras.models import Sequential

    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # Create graph nodes.
    for layer in layers:
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            class_name = '{}({})'.format(class_name, child_class_name)

        if class_name == "InputLayer":
            class_name = "Input"

        # Create node's label.
        label = class_name

        # Add Dense
        try:
            label += " " + str(layer.units)
        except AttributeError:
            # Add Convolutions
            if isinstance(layer, keras.layers.convolutional._Conv):
                kernel_size = "x".join([str(k) for k in layer.kernel_size])
                label += " %s,%s" % (kernel_size, str(layer.filters))

            # Add pool1d
            if isinstance(layer, keras.layers.pooling._Pooling1D):
                label += " " + str(layer.pool_size)

            # Add pool2d
            if isinstance(layer, keras.layers.pooling._Pooling2D):
                pool_size = [str(k) for k in layer.pool_size]
                label += " " + "x".join(pool_size)

        node = pydot.Node(layer_id, label=label)
        node.set("shape", 'box')
        if class_name == "Input":
            node.set("color", 'red')

        if class_name == "Input":
            node.set("color", 'red')

        dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    label = str(inbound_layer.output_shape[1:])
                    edge = pydot.Edge(inbound_layer_id, layer_id, label=label)
                    dot.add_edge(edge)
    return dot


model = applications.VGG16(weights=None)
dot = model_to_dot(model)
tex_code = d2t.dot2tex(dot.to_string(), format='tikz', crop=True)
tex_file = open("model.tex", 'w')
tex_file.write(tex_code)
tex_file.close()
