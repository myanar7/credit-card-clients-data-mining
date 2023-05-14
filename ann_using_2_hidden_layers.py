# Import necessary libraries
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from keras.wrappers.scikit_learn import KerasClassifier

def build_ann_model(input_dim):
    model = tf.keras.Sequential([
        # Add the first hidden layer with 32 neurons and ReLU activation function
        tf.keras.layers.Dense(32, activation='relu', input_dim=input_dim),
        # Add the second hidden layer with 16 neurons and ReLU activation function
        tf.keras.layers.Dense(16, activation='relu'),
        # Add the output layer with 1 neuron and sigmoid activation for binary classification
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model using the Adam optimizer and binary crossentropy loss function
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to perform classification using the ANN model
def ann_2_hidden_layers_classification(data):
    # Prepare the dataset
    X = data.drop('default payment next month', axis=1)
    y = data['default payment next month']

    # Split data into training and test sets using the holdout method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the input features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a function that returns a compiled model
    def create_compiled_model():
        model = build_ann_model(input_dim=X_train.shape[1])
        return model

    # Wrap the model using KerasClassifier
    keras_clf = KerasClassifier(build_fn=create_compiled_model, epochs=100, batch_size=32, verbose=0)

    # Perform cross-validation on the training set
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_train_pred = cross_val_predict(keras_clf, X_train_scaled, y_train, cv=kf)

    # Train the model on the full training set
    keras_clf.fit(X_train_scaled, y_train)

    # Make predictions on the training and test sets
    y_train_pred = (keras_clf.predict(X_train_scaled) > 0.5).astype("int32")
    y_pred = (keras_clf.predict(X_test_scaled) > 0.5).astype("int32")

    return y_train, y_train_pred, y_test, y_pred

# Function to evaluate the performance of the ANN model
def evaluate_ann_2_hidden_layers(y_train, y_train_pred, y_test, y_pred):
    # Calculate confusion matrices for the training and test sets
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    conf_matrix_test = confusion_matrix(y_test, y_pred)
    
    # Calculate accuracy, recall, precision, and F1 scores for the training and test sets
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_pred)
    recall_train = recall_score(y_train, y_train_pred)
    recall_test = recall_score(y_test, y_pred)
    precision_train = precision_score(y_train, y_train_pred)
    precision_test = precision_score(y_test, y_pred)
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_pred)

    # Store the results in a dictionary
    results = {
        'train': {
            'confusion_matrix': conf_matrix_train,
            'accuracy': acc_train,
            'recall': recall_train,
            'precision': precision_train,
            'f1_score': f1_train
        },
        'test': {
            'confusion_matrix': conf_matrix_test,
            'accuracy': acc_test,
            'recall': recall_test,
            'precision': precision_test,
            'f1_score': f1_test
        }
    }

    return results
