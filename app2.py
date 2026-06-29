import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from streamlit_drawable_canvas import st_canvas
import io

st.set_page_config(page_title="Digit Recognizer", page_icon="🔢", layout="centered")

st.title("🔢 Handwritten Digit Recognizer")
st.markdown("**Perceptron vs ANN vs CNN** — MNIST Comparison")
st.markdown("---")

# Model building functions
@st.cache_resource
def build_and_train_models():
    st.info("⏳ Models training ho rahe hain... (first run mein thoda time lagega)")
    
    # Load MNIST
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    X_train_flat = X_train.reshape(-1, 784)
    X_test_flat = X_test.reshape(-1, 784)
    
    # Perceptron
    perceptron = Sequential([
        Dense(10, activation='softmax', input_shape=(784,))
    ])
    perceptron.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    perceptron.fit(X_train_flat, y_train_cat, epochs=3, batch_size=32, verbose=0,
                   validation_split=0.1)
    acc_percp = perceptron.evaluate(X_test_flat, y_test_cat, verbose=0)[1]
    
    # ANN
    ann = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ann.fit(X_train, y_train_cat, epochs=3, batch_size=32, verbose=0,
            validation_split=0.1)
    acc_ann = ann.evaluate(X_test, y_test_cat, verbose=0)[1]
    
    # CNN
    X_train_cnn = X_train.reshape(-1, 28, 28, 1)
    X_test_cnn = X_test.reshape(-1, 28, 28, 1)
    cnn = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train_cnn, y_train_cat, epochs=3, batch_size=32, verbose=0,
            validation_split=0.1)
    acc_cnn = cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)[1]
    
    return perceptron, ann, cnn, acc_percp, acc_ann, acc_cnn

perceptron, ann, cnn, acc_percp, acc_ann, acc_cnn = build_and_train_models()

# Accuracy Cards
st.subheader("📊 Model Accuracy Comparison")
col1, col2, col3 = st.columns(3)
col1.metric("🔵 Perceptron", f"{acc_percp*100:.2f}%")
col2.metric("🟡 ANN", f"{acc_ann*100:.2f}%")
col3.metric("🟢 CNN", f"{acc_cnn*100:.2f}%")

st.markdown("---")
st.subheader("✏️ Apna Digit Draw Karo")

model_choice = st.radio("Konse model se predict karein?", 
                         ["Perceptron", "ANN", "CNN"], horizontal=True)

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("🔍 Predict Karo", use_container_width=True):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        img = img.convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        
        if model_choice == "Perceptron":
            inp = img_array.reshape(1, 784)
            pred = perceptron.predict(inp, verbose=0)
        elif model_choice == "ANN":
            inp = img_array.reshape(1, 28, 28)
            pred = ann.predict(inp, verbose=0)
        else:
            inp = img_array.reshape(1, 28, 28, 1)
            pred = cnn.predict(inp, verbose=0)
        
        predicted_digit = np.argmax(pred)
        confidence = np.max(pred) * 100
        
        st.markdown("---")
        st.markdown(f"## 🎯 Predicted Digit: **{predicted_digit}**")
        st.markdown(f"**Confidence:** {confidence:.1f}%")
        
        st.subheader("All digit probabilities:")
        prob_cols = st.columns(10)
        for i, col in enumerate(prob_cols):
            col.metric(str(i), f"{pred[0][i]*100:.1f}%")
    else:
        st.warning("Pehle canvas pe kuch draw karo!")

st.markdown("---")
st.caption("Built with ❤️ | Perceptron vs ANN vs CNN on MNIST | Learn in Public 🚀")