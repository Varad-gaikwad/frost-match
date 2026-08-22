#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib as jb

#%%
item_features= pd.read_csv('ice_cream_flavors.csv')
user_features= pd.read_csv('user_ratings.csv')

item_features_indexed = item_features.set_index('flavor')#Sets the index to the flavour so the falvours are easy to find.

#%%
def build_user_vector(flavor_rating_pair):
    total_weighted_vector = np.zeros(item_features_indexed.shape[1])
    total_abs_weight = 0

    for f, r in flavor_rating_pair:
        flavor_vec = item_features_indexed.loc[f].values
        weight = r - 3

        total_weighted_vector += flavor_vec * weight
        total_abs_weight += abs(weight)

    if total_abs_weight == 0:
        rating = np.mean(
            [item_features_indexed.loc[f].values
             for f, r in flavor_rating_pair],
            axis=0
        )
    else:
        rating = total_weighted_vector / total_abs_weight

    return rating.astype(np.float32) #As tensorflow is required to have float32 values.


#%%
#Now we have to create user vectors from the user ratings csv.
unique_users= user_features['user_id'].unique()
user_vector={}

for uid in unique_users:
    user_rows = user_features[user_features['user_id'] == uid]
    total_weighted_vector = np.zeros(item_features_indexed.shape[1])
    total_abs_weight = 0

    for index, row in user_rows.iterrows():
        flv = row['flavor']
        rating = row['rating']
        flavor_vec = item_features_indexed.loc[flv].values
        weight = rating - 3
        total_weighted_vector += flavor_vec * weight
        total_abs_weight += abs(weight)

    if total_abs_weight == 0:
        user_vector[uid] = np.mean([item_features_indexed.loc[row['flavor']].values 
                                      for _, row in user_rows.iterrows()], axis=0)
    else:
        user_vector[uid] = total_weighted_vector / total_abs_weight

#Create X_user,X_item and Y to train the model.
X_user_list = []
X_item_list = []
Y_list = []

for index, row in user_features.iterrows():

    uid = row['user_id']
    target_flavor = row['flavor']
    target_rating = row['rating']

    # Get all ratings from this user EXCEPT the flavor
    # that we are trying to predict
    other_ratings = user_features[
        (user_features['user_id'] == uid) &
        (user_features['flavor'] != target_flavor)
    ]

    flavor_rating_pairs = [
        (r['flavor'], r['rating'])
        for _, r in other_ratings.iterrows()
    ]

    # We need at least one other flavor to build the profile
    if len(flavor_rating_pairs) == 0:
        continue

    user_vec = build_user_vector(flavor_rating_pairs)

    item_vec = item_features_indexed.loc[target_flavor].values

    X_user_list.append(user_vec)
    X_item_list.append(item_vec)
    Y_list.append(target_rating)


X_user_train = np.array(X_user_list, dtype=np.float32)
X_item_train = np.array(X_item_list, dtype=np.float32)
y_train_initial = np.array(Y_list, dtype=np.float32)


# %%
#We create a two-towered NN architecture to train user and item embeddings for rating prediction.


user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32)
])
item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32)
])


input_user=tf.keras.layers.Input(shape=(X_user_train.shape[1],))
vu= user_NN(input_user)


input_item=tf.keras.layers.Input(shape=(X_item_train.shape[1],))
vm=item_NN(input_item)


output_layer= tf.keras.layers.Dot(axes=1)
output=output_layer([vu,vm])

model= tf.keras.models.Model([input_user,input_item], output)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error')


# %%

xu_train,xu_test,xi_train,xi_test,y_train,y_test= train_test_split(
    X_user_train,X_item_train,y_train_initial, test_size=0.2, random_state=42
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',     # Monitor validation loss
    patience=10,             # Stop after 5 epochs without improvement ("rounds")
    restore_best_weights=True, # Restore model weights from the best epoch
    verbose=1            
)

history = model.fit(
    [xu_train, xi_train], y_train,
    epochs=50,
    validation_data=([xu_test, xi_test], y_test),
    callbacks=[early_stopping]
)
loss = model.evaluate([xu_test, xi_test], y_test)
print(f"Test MSE: {loss:.4f}")



# %%
def recommend_flavors(model,flavor_ratings_pairs, top_n=3):
    new_user_vector= build_user_vector(flavor_ratings_pairs)

    flavors_list=[]

    for flavor,rating in flavor_ratings_pairs:
        flavors_list.append(flavor)#Extracts all the flavors from the pairs

    all_flavors = item_features_indexed.index.tolist()
    #Converts all the flavours into a list

    candidate=[]

    for flavor in all_flavors:
        if flavor not in flavors_list:
            candidate.append(flavor)

    # We need the SAME user vector repeated once for every candidate flavor,
    # because the model expects one (user, item) pair at a time.

    user_input=[]
    for i in range(len(candidate)):
        user_input.append(new_user_vector)
    user_input=np.array(user_input, dtype=np.float32)

    item_input=[]
    for flavor in candidate:
        item_input.append(item_features_indexed.loc[flavor].values)
    item_input=np.array(item_input,dtype=np.float32)

    predictions=model.predict([user_input,item_input])
    predictions=predictions.flatten()

    #TO CONVERT THE MODEL FROM RAW OUTPUT TO 1-5 RANGE
    min_prediction = predictions.min()
    max_prediction = predictions.max()

    if max_prediction != min_prediction:
         predictions = 1 + 4 * ((predictions - min_prediction)/ (max_prediction - min_prediction))
    else: #If max_prediction == min prediction, all predicitions are equal and then all are set to 3.
        predictions = np.full_like(
        predictions,
        3.0,
        dtype=np.float32
    )


    #Pair up each flavor with its predicted rating
    results= list(zip(candidate,predictions))

    results.sort(key= lambda pair: pair[1], reverse=True)
    '''key=lambda pair: pair[1] — tells .sort() "when comparing two items to decide their order,
      don't look at the whole tuple — only look at position 1 (the rating), 
      ignore position 0 (the flavor name)"
    lambda is a keyword to create a temporary function
    pair: The input parameter representing a single item from results'''

    return results[:top_n]

model.save('ice_cream_model.keras')
jb.dump(item_features_indexed,'item_features_indexed.pkl')

all_train_predictions = model.predict([X_user_train, X_item_train]).flatten()
GLOBAL_MIN = all_train_predictions.min()
GLOBAL_MAX = all_train_predictions.max()

jb.dump((GLOBAL_MIN, GLOBAL_MAX), 'prediction_range.pkl')

print('MODELS TRAINED AND SAVED SUCCESSFULLY.')




     
# %%
