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

        dot.add_node(node)

    # add output node
    output_node = pydot.Node("output_node", label="Output")
    output_node.set("shape", 'box')
    dot.add_node(output_node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model._network_nodes:
                inbound_layers = node.inbound_layers
                if not isinstance(inbound_layers, list):
                    inbound_layers = [inbound_layers]
                for inbound_layer in inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    output_shape = inbound_layer.output_shape
                    if isinstance(output_shape, list):
                        if len(output_shape) > 1:
                            raise Exception("More than one output_shape found")
                        output_shape = output_shape[0]
                    label = str(output_shape[1:])
                    edge = pydot.Edge(inbound_layer_id, layer_id, label=label)
                    dot.add_edge(edge)

    # connect output
    out_edge = pydot.Edge(
        str(id(layers[-1])), "output_node", label=str(model.output_shape[1:])
    )
    dot.add_edge(out_edge)

    return dot


def gen_tikz_from_model(model):
    dot = model_to_dot(model)
    return d2t.dot2tex(dot.to_string(), format='tikz', crop=True)


if __name__ == '__main__':
    from keras import applications
    model = applications.VGG16(weights=None)
    tix_code = gen_tikz_from_model(model)
    tex_file = open("model.tex", 'w')
    tex_file.write(tex_file)
    tex_file.close()
