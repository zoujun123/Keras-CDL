import numpy as np
from keras.layers import Input, Embedding, Dot, Flatten, Dense, Dropout
from keras.layers.noise import GaussianNoise
from keras.initializers import RandomUniform
from keras.models import Model
from keras.regularizers import l2
from keras import optimizers
from keras import backend as K


class CollaborativeDeepLearning:
    def __init__(self, num_user, hidden_layers):
        self.num_user = num_user
        self.hidden_layers = hidden_layers
        self.item_dim = hidden_layers[0]
        self.embedding_dim = hidden_layers[-1]
        
    def pretrain(self, X_train, encoder_noise=0.1, dropout_rate=0.1, activation='sigmoid', batch_size=64, nb_epoch=10):
        '''
        Layer-wise pre-training
        X_train = item features matrix
        '''
        self.trained_encoders = []
        self.trained_decoders = []
        X_train_tmp = X_train
        for input_dim, hidden_dim in zip(self.hidden_layers[:-1], self.hidden_layers[1:]):
            print('Pre-training the layer: Input dim {} -> Output dim {}'.format(input_dim, hidden_dim))
            pretrain_input = Input(shape=(input_dim,))
            encoded = GaussianNoise(stddev=encoder_noise)(pretrain_input)
            encoded = Dropout(dropout_rate)(encoded)
            encoder = Dense(hidden_dim, activation=activation)(encoded)
            decoder = Dense(input_dim, activation=activation)(encoder)
            # end to end ae
            ae = Model(inputs=pretrain_input, outputs=decoder)
            # encoder
            ae_encoder = Model(inputs=pretrain_input, outputs=encoder)
            # decoder
            encoded_input = Input(shape=(hidden_dim,))
            decoder_layer = ae.layers[-1] # the last layer
            ae_decoder = Model(encoded_input, decoder_layer(encoded_input))

            ae.compile(loss='mse', optimizer='rmsprop')
            ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, epochs=nb_epoch)

            self.trained_encoders.append(ae_encoder)
            self.trained_decoders.append(ae_decoder)
            X_train_tmp = ae_encoder.predict(X_train_tmp)

    def fineture(self, train_mat, test_mat, item_mat, lr=0.1, reg=0.1, epochs=10, batch_size=64):
        '''
        Fine-tuning with 
        '''
        item_input = Input(shape=(self.item_dim,))

        encoded = self.trained_encoders[0](item_input)
        encoded = self.trained_encoders[1](encoded)

        decoded = self.trained_decoders[1](encoded)
        decoded = self.trained_decoders[0](decoded)

        userInputLayer = Input(shape=(1,), dtype='int32')
        userEmbeddingLayer = Embedding(input_dim=self.num_user, output_dim=self.embedding_dim, input_length=1, embeddings_regularizer=l2(reg), embeddings_initializer=RandomUniform(minval=0, maxval=1))(userInputLayer)
        userEmbeddingLayer = Flatten()(userEmbeddingLayer)

        dotLayer = Dot(axes = -1)([userEmbeddingLayer, encoded])
        cdl = Model(inputs=[userInputLayer, item_input], outputs=dotLayer)

        # sgd = optimizers.SGD(lr=lr, decay=0, momentum=0.9, nesterov=False)
        cdl.compile(optimizer='adam', loss='mse')

        train_user, train_item_feat, train_label = self.matrix2input(train_mat, item_mat)
        test_user, test_item_feat, test_label = self.matrix2input(test_mat, item_mat)

        model_history = cdl.fit([train_user, train_item_feat], train_label, epochs=epochs, batch_size=batch_size, validation_data=([test_user, test_item_feat], test_label))
        return model_history

    def matrix2input(self, rating_mat, item_mat):
        train_user = rating_mat[:, 0].reshape(-1, 1).astype(int)
        train_item = rating_mat[:, 1].reshape(-1, 1).astype(int)
        train_label = rating_mat[:, 2].reshape(-1, 1)
        train_item_feat = [item_mat[train_item[x]][0] for x in range(train_item.shape[0])]
        return train_user, np.array(train_item_feat), train_label
