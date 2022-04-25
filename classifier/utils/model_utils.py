import tensorflow.keras.layers as layers


def add_layer(x, ch, drop, architecture, arch_params):
    """ 
    Passes the input tensor layer to the model according to the arcitechture specified
    """
    if architecture == 'conv_1D':
        conv_size = arch_params.conv_size
        x = conv_bn_activate(x, ch, conv_size=conv_size, drop=drop)
    elif architecture == 'attention':
        x = attention_bn_activate(x, ch, query_dim=ch, drop=drop)
    elif architecture == 'lstm':
        x = dense_bn_activate(x, ch, drop=drop)  # LSTM also uses fc except first layer
    elif architecture == 'fully_connected':
        x = dense_bn_activate(x, ch, drop=drop)
    else:
        raise NotImplementedError
    return x


def dense_bn_activate(x, out_dim, activation='relu', drop=0.):
    """ 
    Fully Connected -> BatchNormalization -> Activation -> Dropout
    """
    x = layers.Dense(out_dim)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if drop > 0:
        x = layers.Dropout(rate=drop)(x)
    return x


def conv_bn_activate(x, out_dim, activation='relu', conv_size=3, drop=0.):
    """ 
    1D Convolution -> BatchNormalization -> Activation -> MaxPool -> Dropout
    """
    x = layers.Conv1D(out_dim, conv_size)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, 2)(x)
    if drop > 0:
        x = layers.Dropout(rate=drop)(x)
    return x


def attention(x, query_dim, out_dim):
    """ 
    Q K V attention accross the time axis, where Q = K
    """
    q = layers.Conv1D(query_dim, 1)(x)
    v = layers.Conv1D(query_dim, 1)(x)
    attn = layers.Attention()([q, v])
    outs = layers.Conv1D(out_dim, 1)(attn)
    return outs


def attention_bn_activate(x, out_dim, query_dim, activation='relu', drop=0.):
    """ 
    Attention -> BatchNormalization -> Activation -> MaxPool -> Dropout
    """
    x = attention(x, query_dim, out_dim)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPooling1D(2, 2)(x)
    if drop > 0:
        x = layers.Dropout(rate=drop)(x)
    return x


def freeze_model_except_last_layer(model):
    """ 
    Set all layers except top layer to non trainable
    """
    for idx, layer in enumerate(model.layers[:-1]):
        if not isinstance(layer, layers.BatchNormalization):
            model.layers[idx].trainable = False


def unfreeze_model_except_last_layer(model):
    """ 
    Set all to trainable
    """
    for idx, layer in enumerate(model.layers[:-1]):
        if not isinstance(layer, layers.BatchNormalization):
            model.layers[idx].trainable = True


def copy_model_weights_except_last_layer(target_model, source_model):
    """ 
    Copy model weights for transfer learning
    """
    for idx in range(len(source_model.layers[:-1])):
        weights = source_model.layers[idx].get_weights()
        target_model.layers[idx].set_weights(weights)
