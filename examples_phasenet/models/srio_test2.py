from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense


def convnet(input_shape=(16,32,32, 1), output_size=11, kernel_size=(3,3,3), activation='tanh', padding='same', pool_size=(1,2,2)):
    print(">>> input_shape", input_shape)
    print(">>> output_size", output_size)
    print(">>> kernel_size", kernel_size)
    print(">>> activation", activation)
    print(">>> padding", padding)
    print(">>> pool_size", pool_size)

    inp = Input(name='X', shape=input_shape)
    t = Conv3D(8, name='conv1', kernel_size=kernel_size, activation=activation, padding=padding)(inp)
    t = Conv3D(8, name='conv2', kernel_size=kernel_size, activation=activation, padding=padding)(t)
    t = MaxPooling3D(name='maxpool1', pool_size=pool_size)(t)
    t = Conv3D(16, name='conv3', kernel_size=kernel_size, activation=activation, padding=padding)(t)
    t = Conv3D(16, name='conv4', kernel_size=kernel_size, activation=activation, padding=padding)(t)
    t = MaxPooling3D(name='maxpool2', pool_size=pool_size)(t)
    t = Conv3D(32, name='conv5', kernel_size=kernel_size, activation=activation, padding=padding)(t)
    t = Conv3D(32, name='conv6', kernel_size=kernel_size, activation=activation, padding=padding)(t)
    t = MaxPooling3D(name='maxpool3', pool_size=pool_size)(t)
    t = Conv3D(64, name='conv7', kernel_size=kernel_size, activation=activation, padding=padding)(t)
    t = Conv3D(64, name='conv8', kernel_size=kernel_size, activation=activation, padding=padding)(t)
    t = MaxPooling3D(name='maxpool4', pool_size=pool_size)(t)
    t = Conv3D(128, name='conv9', kernel_size=kernel_size, activation=activation, padding=padding)(t)
    t = Conv3D(128, name='conv10', kernel_size=kernel_size, activation=activation, padding=padding)(t)

    if input_shape[0] == 1:
        t = MaxPooling3D(name='maxpool5', pool_size=(1, 2, 2))(t)
    else:
        t = MaxPooling3D(name='maxpool5', pool_size=(2, 2, 2))(t)
    t = Flatten(name='flat')(t)
    t = Dense(64, name='dense1', activation=activation)(t)
    t = Dense(64, name='dense2', activation=activation)(t)

    oup = Dense(output_size, name='Y', activation='linear')(t)

    return inp, oup


if __name__ == "__main__":
    from tensorflow.keras import models
    from tensorflow.keras.models import Model

    inp, out = convnet()
    m = Model(inp, out)
    # model = models.Sequential(out)

    # model.add(out)
    print(m.summary())

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 X (InputLayer)              [(None, 16, 32, 32, 1)]   0         
                                                                 
 conv1 (Conv3D)              (None, 16, 32, 32, 8)     224       
                                                                 
 conv2 (Conv3D)              (None, 16, 32, 32, 8)     1736      
                                                                 
 maxpool1 (MaxPooling3D)     (None, 16, 16, 16, 8)     0         
                                                                 
 conv3 (Conv3D)              (None, 16, 16, 16, 16)    3472      
                                                                 
 conv4 (Conv3D)              (None, 16, 16, 16, 16)    6928      
                                                                 
 maxpool2 (MaxPooling3D)     (None, 16, 8, 8, 16)      0         
                                                                 
 conv5 (Conv3D)              (None, 16, 8, 8, 32)      13856     
                                                                 
 conv6 (Conv3D)              (None, 16, 8, 8, 32)      27680     
                                                                 
 maxpool3 (MaxPooling3D)     (None, 16, 4, 4, 32)      0         
                                                                 
 conv7 (Conv3D)              (None, 16, 4, 4, 64)      55360     
                                                                 
 conv8 (Conv3D)              (None, 16, 4, 4, 64)      110656    
                                                                 
 maxpool4 (MaxPooling3D)     (None, 16, 2, 2, 64)      0         
                                                                 
 conv9 (Conv3D)              (None, 16, 2, 2, 128)     221312    
                                                                 
 conv10 (Conv3D)             (None, 16, 2, 2, 128)     442496    
                                                                 
 maxpool5 (MaxPooling3D)     (None, 8, 1, 1, 128)      0         
                                                                 
 flat (Flatten)              (None, 1024)              0         
                                                                 
 dense1 (Dense)              (None, 64)                65600     
                                                                 
 dense2 (Dense)              (None, 64)                4160      
                                                                 
 Y (Dense)                   (None, 11)                715           
                                                                 
=================================================================
Total params: 954,195
Trainable params: 954,195
Non-trainable params: 0
_________________________________________________________________

"""
