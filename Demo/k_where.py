#dimension
max_token_index=1246
max_len=419

d=64
from keras.models import Model
from keras.layers import Input, GRU, Dense ,Flatten , Concatenate,Bidirectional
from keras.layers.embeddings import Embedding
from keras.constraints import max_norm
from keras import regularizers

def get_k_where_model():
    # Define an input sequence and process it.
    question_input = Input(shape=(max_len,),name='Q_input')

    embedding= Embedding(max_token_index, d, input_length=max_len,name='embedding')
    # embedding_C= Embedding(max_token_index, d, input_length=max_len,name='embedding_C')
    #                      embeddings_constraint=max_norm(2.),
    #                      embeddings_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))

    Q_embedding= embedding(question_input)



    encoder_question = Bidirectional(GRU(d, return_state=True))
    _ , Q_state_h1, Q_state_h2 = encoder_question(Q_embedding)


    con=Concatenate()([Q_state_h1,Q_state_h2])


    final=Dense(5,activation='softmax')(con)

    model = Model(question_input, final)
    model.load_weights('where_k_best_model.h5')
    return model