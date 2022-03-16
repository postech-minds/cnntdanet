from tensorflow import keras


def get_cnn_tda_net(method, input_shape, n_classes):
    local_pipeline = get_2d_cnn(input_shape['local'], name='local_pipeline')
    if method == 'persistence-image':
        global_pipeline = get_2d_cnn(input_shape['global'], name='global_pipeline')
    elif method in ['betti-curve', 'persistence-landscape']:
        global_pipeline = get_1d_cnn(input_shape['global'], name='global_pipeline')
    tail = get_tail(n_classes)

    input_image = keras.layers.Input(input_shape['local'])
    input_tda   = keras.layers.Input(input_shape['global'])

    local_net  = local_pipeline(input_image)
    global_net = global_pipeline(input_tda)
    concat = keras.layers.concatenate([local_net, global_net])
    out = tail(concat)

    return keras.Model(inputs=[input_image, input_tda], outputs=[out])


def get_cnn_net(input_shape, n_classes):
    local_pipeline = get_2d_cnn(input_shape, name='local_pipeline')
    tail = get_tail(n_classes)

    return keras.models.Sequential([local_pipeline, tail])


def get_1d_cnn(input_shape, name='global_pipeline'):
    cnn = keras.models.Sequential(name=name)
    cnn.add(keras.layers.InputLayer(input_shape=input_shape))
    for rate in (1, 2, 4, 8) * 2:
        cnn.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding='causal', activation='relu', dilation_rate=rate))
    cnn.add(keras.layers.Conv1D(filters=10, kernel_size=1))
    cnn.add(keras.layers.Flatten())

    return cnn


def get_2d_cnn(input_shape, name='local_pipeline'):
    cnn = keras.models.Sequential([
        keras.layers.Conv2D(16, 3, activation="relu", padding='same', input_shape=input_shape),
        keras.layers.Conv2D(32, 3, activation="relu", padding='same'),
        keras.layers.AveragePooling2D(2),
        keras.layers.Conv2D(64 ,3, activation="relu", padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, 3, activation="relu", padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten()
    ], name=name)

    return cnn


def get_tail(n_classes, name='tail'):
    tail = keras.models.Sequential([
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(n_classes, activation='softmax')
    ], name=name)

    return tail
