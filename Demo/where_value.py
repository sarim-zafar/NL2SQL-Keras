#dimension
max_len=419
max_token_index=1246

#dimension
d=64
from keras.models import Model
from keras.layers import Input, GRU, Dense ,Flatten , Concatenate,Bidirectional,Average,RepeatVector,Flatten
from keras.layers.embeddings import Embedding
from keras.constraints import max_norm
from keras import regularizers


def get_where_value_model():
    # Define an input sequence and process it.
    question_input = Input(shape=(max_len,),name='Q_input')
    column_input = Input(shape=(max_len,),name='C_input')
    ops_input = Input(shape=(3,),name='Ops_input')

    embedding= Embedding(max_token_index, d, input_length=max_len,name='embedding')
    # embedding_C= Embedding(max_token_index, d, input_length=max_len,name='embedding_C')
    #                      embeddings_constraint=max_norm(2.),
    #                      embeddings_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))

    Q_embedding= embedding(question_input)
    C_embedding= embedding(column_input)



    encoder_question = Bidirectional(GRU(d, return_state=True))
    Q_act , Q_state_h1, Q_state_h2 = encoder_question(Q_embedding)

    encoder_column = Bidirectional(GRU(d, return_state=True))
    C_act , C_state_h1, C_state_h2 = encoder_column(C_embedding)


    con=Concatenate()([Q_act,Q_state_h1,Q_state_h2,C_act,C_state_h1,C_state_h2,ops_input])


    start=Dense(max_len,activation='softmax',name='start')(con)

    con_2=Concatenate()([con,start])

    end=Dense(max_len,activation='softmax',name='end')(con_2)

    model = Model([question_input, column_input,ops_input], [start,end])
    model.load_weights('where_value_best_model.h5')
    return model