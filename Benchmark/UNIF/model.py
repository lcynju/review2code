import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses


def get_model(longer_code, longer_desc, number_code_tokens, number_desc_tokens):
    embedding_size = 1000

    code_input = tf.keras.Input(shape=(longer_code,), name="code_input")
    code_embeding = tf.keras.layers.Embedding(number_code_tokens, embedding_size, name="code_embeding")(code_input)

    attention_code = tf.keras.layers.Attention(name="attention_code")([code_embeding, code_embeding])

    query_input = tf.keras.Input(shape=(longer_desc,), name="query_input")
    query_embeding = tf.keras.layers.Embedding(number_desc_tokens, embedding_size, name="query_embeding")(query_input)


    code_output = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1), name="sum")( attention_code)
    query_output = tf.keras.layers.GlobalAveragePooling1D(name="average")( query_embeding)

    # This model generates code embedding
    model_code = tf.keras.Model(inputs=[code_input], outputs=[code_output], name='model_code')
    # This model generates description/query embedding
    model_query = tf.keras.Model(inputs=[query_input], outputs=[query_output], name='model_query')

    # Cosine similarity
    # If normalize set to True, then the output of the dot product is the cosine proximity between the two samples.
    cos_sim = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_sim')([query_output, code_output])
    # This model calculates cosine similarity between code and query pairs
    cos_model = tf.keras.Model(inputs=[code_input, query_input], outputs=[cos_sim],name='sim_model')

    loss = tf.keras.layers.Flatten()(cos_sim)
    training_model = tf.keras.Model(inputs=[code_input, query_input], outputs=[cos_sim],name='training_model')

    # model_code.compile(loss=losses.cosine_proximity, optimizer='adam')
    model_code.compile(loss='cosine_proximity', optimizer='adam')
    # model_query.compile(loss=losses.cosine_proximity, optimizer='adam')
    model_query.compile(loss='cosine_proximity', optimizer='adam')

    cos_model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=["accuracy"])  # extract similarity

    # Negative sampling
    good_desc_input = tf.keras.Input(shape=(longer_desc,), name="good_desc_input")
    bad_desc_input = tf.keras.Input(shape=(longer_desc,), name="bad_desc_input")

    good_desc_output = cos_model([code_input, good_desc_input])
    bad_desc_output = cos_model([code_input, bad_desc_input])

    margin = 0.9
    loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, margin - x[0] + x[1]), output_shape=lambda x: x[0], name='loss')([good_desc_output, bad_desc_output])
    training_model = tf.keras.Model(inputs=[code_input, good_desc_input, bad_desc_input], outputs=[loss],name='training_model')

    training_model.compile(loss=lambda y_true, y_pred:  y_pred+y_true-y_true, optimizer='adam')
    # y_true-y_true avoids warning

    # tf.keras.utils.plot_model(cos_model, "cos_model.png", show_shapes=True)
    return training_model, cos_model, model_code, model_query