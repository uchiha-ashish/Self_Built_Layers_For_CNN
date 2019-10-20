def inception_block(X, filters, stage, block):
    """  
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_path1 = X
    X_path2 = X
    X_path3 = X
    X_shortcut = X
    
    # First component of main path
    X_path1 = Conv2D(filters = F1, kernel_size = (1, 1), padding = 'same', name = conv_name_base + '2a')(X_path1)
    X_path1 = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X_path1)

    # Second component of main path
    X_path2 = Conv2D(filters = F2, kernel_size = (3, 3), padding = 'same', name = conv_name_base + '2b')(X_path2)
    X_path2 = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X_path2)

    # Third component of main path
    X_path3 = Conv2D(filters = F3, kernel_size = (5, 5), padding = 'same', name = conv_name_base + '2c')(X_path3)
    X_path3 = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X_path3)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = layers.Concatenate(axis = -1)([X_path1, X_path2, X_path3, X_shortcut])
    X = Activation('relu')(X)
    
    return X
    
    
 
 def DeceptiNet(input_shape = (224,224,3), classes = 185):
     
    X_input = Input(input_shape)
#   X = ZeroPadding2D((16, 16))(X_input)

    X = Conv2D(8, (7, 7), padding = 'same', name = 'conv0', kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = Conv2D(32, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((7, 7), strides=(2, 2))(X)
    
    X = inception_block(X, [32,32,32], 2, 'inception_1')
    
    X = Conv2D(256, (7, 7), strides = (2,2), name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    X = Conv2D(512, (7, 7), name = 'conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv3')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((3, 3), strides=(2, 2))(X)
    
    X = Flatten()(X)
    X = Dense(2048, activation='relu', name='fc-1')(X)
  #  X = Dropout(0.2)(X)
    X = Dense(1024, activation='relu', name='fc-2')(X)
   # X = Dropout(0.2)(X)
    X = Dense(512, activation='relu', name='fc-3')(X)
    X = Dense(256, activation='relu', name='fc-4')(X)
# X = Dropout(0.2)(X)
    #     X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = Model(inputs = X_input, outputs = X, name='DeceptiNet')
    return model
