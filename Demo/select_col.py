#dimension
max_len=419
max_token_index=1246

from keras import backend as K
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


d=64
from keras.models import Model
from keras.layers import Input, GRU, Dense ,Flatten , Concatenate,Bidirectional,Lambda,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.constraints import max_norm
from keras import regularizers

def get_col_model():
    # Define an input sequence and process it.
    question_input = Input(shape=(max_len,),name='Q_input')
    column_input = Input(shape=(max_len,),name='C_input')

    embedding= Embedding(max_token_index, d, input_length=max_len,name='embedding')
    # embedding_C= Embedding(max_token_index, d, input_length=max_len,name='embedding_C')
    #                      embeddings_constraint=max_norm(2.),
    #                      embeddings_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))

    Q_embedding= embedding(question_input)
    C_embedding= embedding(column_input)




    encoder_question = Bidirectional(GRU(d, return_state=True))
    _ , Q_state_h1, Q_state_h2 = encoder_question(Q_embedding)

    encoder_column = Bidirectional(GRU(d, return_state=True))
    _ , C_state_h1, C_state_h2 = encoder_column(C_embedding)


    con_Q=Concatenate()([Q_state_h1,Q_state_h2])
    con_C=Concatenate()([C_state_h1,C_state_h2])

    final = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([con_Q, con_C])
    # final=Dense(1,activation='sigmoid')(con)

    model = Model([question_input, column_input], final)

    model.load_weights('select_col_best_model.h5')
    return model