import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import time
from sklearn import preprocessing
import warnings

print("Tensorflow Version:\t"+tf.__version__)

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "BTC-USD"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-RATIO-{RATIO_TO_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocess_df(df_preprocess):
    df_preprocess = df_preprocess.drop("future", 1)

    for col in df.columns:
        if col != "target":
            df_preprocess[col] = df_preprocess[col].pct_change()
            df_preprocess.dropna(inplace=True)
            df_preprocess[col] = preprocessing.scale(df_preprocess[col].values)

    df_preprocess.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df_preprocess.values:
        prev_days.append([n for n in i[:-1]])

        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    # Part 3
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    # balancimi i te dhenave
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    x = []
    y = []

    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)

    return np.array(x), y


main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
for ratio in ratios:
    print(ratio)
    dataset = f'crypto_data/{ratio}.csv'
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])

    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True)

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

main_df.dropna(inplace=True)
# print(main_df.head())


# Part 2
times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

# print(last_5pct)

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print("\n\nIMPORTANT DATA")
print(f"Train data:\t {len(train_x)} \nValidation:\t {len(validation_x)}")
print(f"\nDon't buy:\t {train_y.count(0)}, \nBuy:\t\t {train_y.count(1)}")
print(f"\nValidation don't buy:\t {validation_y.count(0)}, \nBuys:\t\t\t\t\t {validation_y.count(1)}")

# preprocess_df(main_df)

# Part 4
model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"

checkpoint = ModelCheckpoint("models/{}.model".format(
    filepath,
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='max'
))

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Train model
history = model.fit(
    np.asarray(train_x), np.asarray(train_y),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    #callbacks=[tensorboard, checkpoint]
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save model
model.save("models/{}".format(NAME))

# warnings.filterwarnings("ignore", category=DeprecationWarning)

