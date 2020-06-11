# Transfer Model Utils
from transfer_model_utils import *

orders_df = pd.read_csv('data/orders_df.csv', index_col=0)

img_dir = 'images' # folder containing all other folders of images
paths = orders_df['file_path']
bucket = 'cwbirdsimages'

def resize_images_array(img_dir, file_paths):
    # arrays of image pixels
    img_arrays = []
    
    # loop through the dataframe that is linked to its label so that all images are in the same order
    for path in tqdm(file_paths):
        s3 = boto3.client('s3')
        try:
            obj = s3.get_object(Bucket=bucket, Key=f'{img_dir}/{path}')
            img_bytes = BytesIO(obj['Body'].read())
            open_img = Image.open(img_bytes)
            arr = np.array(open_img.resize((299,299))) # (299,299) required for Xception
            img_arrays.append(arr)
        except:
            # print(path)
            continue
    return np.array(img_arrays)

# obtain image data in arrays
X = resize_images_array(img_dir, orders_df['file_path'][:21129])

# normalize RGB values
X = X/255.0

# grab label
# INPUT VALUES MUST BE ARRAYS
label = np.array(orders_df['species_group'][:21129].values)

# labels are alphabetical with np.unique
y = (label.reshape(-1,1) == np.unique(orders_df['species_group'][:21129])).astype(float)

# number of outputs/labels available and image input size
n_categories = y.shape[1]
input_size = (299,299,3)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Set tensorboard callback with specified folder and timestamp
tensorboard_callback = TensorBoard(log_dir='logs/', histogram_freq=1)

# create transfer model
transfer_model = create_transfer_model((299,299,3),n_categories)

# change new head to the only trainable layers
_ = change_trainable_layers(transfer_model, 132)

# compile model
transfer_model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# fit model
history = transfer_model.fit(X_train, y_train, batch_size=1000, epochs=15, validation_split=0.1, callbacks=[tensorboard_callback])

transfer_model.save('saved_models/species3_xception.h5')

print('Model saved.')
# load_L_xception = tf.keras.models.load_model('saved_models/large_xception.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

df = pd.DataFrame(acc, columns=['accuracy'])
df['val_accuracy'] = val_acc
df['loss'] = loss
df['val_loss'] = val_loss

df.to_csv('data/accuracy.csv')
print('Accuracy CSV saved.')

pred_prob = transfer_model.predict(X_test)
print('X_test predicted')

pred_arr = []

for i in pred_prob:
    i[i.argmax()] = 1
    i[i < 1] = 0
    print(i)
    pred_arr.append(i)
    
pred_arr = np.array(pred_arr)

print('Starting SKLEARN CLASSIFICATION REPORT')
sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=pred_arr)
print(sk_report)
np.save("data/sk_report.npy", sk_report)
print('sk_report saved.')

print('Begin custom CLASS REPORT')
report_with_auc = class_report(
    y_true=y_test, 
    y_pred=pred_arr)
print('report variable created')
print(report_with_auc)
report_with_auc.to_csv('data/class_report_xception.csv')
print('report saved.')

print('Starting Confusion Matrix...')
conf_mat = confusion_matrix(y_test.argmax(axis=1), pred_arr.argmax(axis=1))
np.savetxt('data/confusion_matrix.csv', conf_mat)

print('Starting recall score...')
recall = recall_score(y_test.argmax(axis=1),pred_arr.argmax(axis=1), average='micro')
print('recall variable obtained.')
np.save("data/recall.npy", recall)

print('Onto Classification Report...')
classify = classification_report(y_test.argmax(axis=1), pred_arr.argmax(axis=1))
print('classify variable obtained.')
np.save("data/classify.npy", classify)

# print('Starting ROC Curve Plot')

# fpr, tpr, thresholds = roc_curve(y_test, pred_arr)
# fig, ax = plt.subplots(figsize=(8,6))
# auc_score = metrics.roc_auc_score(y_test, pred_arr)
# plot_roc_curve(ax, fpr, tpr, auc_score,'Xception ROC Curve')
# plt.savefig('graphs/test_xception_roc_curve.png')
# print('ROC saved')

print('End.')