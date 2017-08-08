import numpy as np
import logging
from keras.layers import Input, Embedding, Dot, Flatten, Dense, Dropout
from keras.layers.noise import GaussianNoise
from keras.initializers import RandomUniform
from keras.models import Model
from keras.regularizers import l2
from keras import optimizers
from keras import backend as K

class CollaborativeDeepLearning:
    def __init__(self, item_mat, hidden_layers):
        '''
        hidden_layers = a list of three integer indicating the embedding dimension of autoencoder
        item_mat = item feature matrix with shape (# of item, # of item features)
        '''
        self.item_mat = item_mat
        self.hidden_layers = hidden_layers
        self.item_dim = hidden_layers[0]
        self.embedding_dim = hidden_layers[-1]
        
    def pretrain(self, lamda_w=0.1, encoder_noise=0.1, dropout_rate=0.1, activation='sigmoid', batch_size=32, epochs=10):
        '''
        layer-wise pretraining on item features (item_mat)
        '''
        self.trained_encoders = []
        self.trained_decoders = []
        X_train = self.item_mat
        first_layer = True
        for input_dim, hidden_dim in zip(self.hidden_layers[:-1], self.hidden_layers[1:]):
            logging.info('Pretraining the layer: Input dim {} -> Output dim {}'.format(input_dim, hidden_dim))
            pretrain_input = Input(shape=(input_dim,))
            if first_layer: # get the corrupted input x_0 from the clean input x_c
                first_layer = False
                encoded = GaussianNoise(stddev=encoder_noise)(pretrain_input)
                encoded = Dropout(dropout_rate)(encoded)
            else:
                encoded = Dropout(dropout_rate)(pretrain_input)
            encoder = Dense(hidden_dim, activation=activation, kernel_regularizer=l2(lamda_w), bias_regularizer=l2(lamda_w))(encoded)
            decoder = Dense(input_dim, activation=activation, kernel_regularizer=l2(lamda_w), bias_regularizer=l2(lamda_w))(encoder)
            # end to end autoencoder
            ae = Model(inputs=pretrain_input, outputs=decoder)
            # encoder
            ae_encoder = Model(inputs=pretrain_input, outputs=encoder)
            # decoder
            encoded_input = Input(shape=(hidden_dim,))
            decoder_layer = ae.layers[-1] # the last layer
            ae_decoder = Model(encoded_input, decoder_layer(encoded_input))

            ae.compile(loss='mse', optimizer='rmsprop')
            ae.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, verbose=2)

            self.trained_encoders.append(ae_encoder)
            self.trained_decoders.append(ae_decoder)
            X_train = ae_encoder.predict(X_train)

    def fineture(self, train_mat, test_mat, lamda_u=0.1, lamda_v=0.1, lamda_n=0.1, lr=0.001, batch_size=64, epochs=10):
        '''
        Fine-tuning with rating prediction
        '''
        num_user = int( max(train_mat[:,0].max(), test_mat[:,0].max()) + 1 )
        #num_user = int( train_mat[:,0].max() + 1 )

        # item autoencoder 
        item_input = Input(shape=(self.item_dim,), name='item_input')
        encoded = self.trained_encoders[0](item_input)
        encoded = self.trained_encoders[1](encoded)
        decoded = self.trained_decoders[1](encoded)
        decoded = self.trained_decoders[0](decoded)

        # item embedding
        itemEmbeddingLayer = GaussianNoise(stddev=lamda_v)(encoded)

        # user embedding
        userInputLayer = Input(shape=(1,), dtype='int32', name='user_input')
        userEmbeddingLayer = Embedding(input_dim=num_user, output_dim=self.embedding_dim, input_length=1, embeddings_regularizer=l2(lamda_u), embeddings_initializer=RandomUniform(minval=0, maxval=1))(userInputLayer)
        userEmbeddingLayer = Flatten()(userEmbeddingLayer)

        # rating prediction
        dotLayer = Dot(axes = -1, name='dot_layer')([userEmbeddingLayer, itemEmbeddingLayer])

        my_RMSprop = optimizers.RMSprop(lr=lr)

        self.cdl_model = Model(inputs=[userInputLayer, item_input], outputs=[dotLayer, decoded])
        self.cdl_model.compile(optimizer=my_RMSprop, loss=['mse', 'mse'], loss_weights=[1, lamda_n])

        train_user, train_item_feat, train_label = self.matrix2input(train_mat)
        test_user, test_item_feat, test_label = self.matrix2input(test_mat)

        model_history = self.cdl_model.fit([train_user, train_item_feat], [train_label, train_item_feat], epochs=epochs, batch_size=batch_size, validation_data=([test_user, test_item_feat], [test_label, test_item_feat]))
        return model_history

    def matrix2input(self, rating_mat):
        train_user = rating_mat[:, 0].reshape(-1, 1).astype(int)
        train_item = rating_mat[:, 1].reshape(-1, 1).astype(int)
        train_label = rating_mat[:, 2].reshape(-1, 1)
        train_item_feat = [self.item_mat[train_item[x]][0] for x in range(train_item.shape[0])]
        return train_user, np.array(train_item_feat), train_label
    
    def getRMSE(self, test_mat):
        test_user, test_item_feat, test_label = self.matrix2input(test_mat)
        pred_out = self.cdl_model.predict([test_user, test_item_feat])
        return np.sqrt(np.mean(np.square(test_label.flatten() - pred_out[0].flatten())))
