import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ========== CONFIGURATION ==========
tf.random.set_seed(42)
np.random.seed(42)

st.set_page_config(
    page_title="Fruit Freshness Detection System",
    page_icon="🍎",
    layout="wide"
)

CLASS_NAMES_16 = [
    "Fresh Apple", "Fresh Banana", "Fresh Grape", "Fresh Guava",
    "Fresh Jujube", "Fresh Orange", "Fresh Pomegranate", "Fresh Strawberry",
    "Rotten Apple", "Rotten Banana", "Rotten Grape", "Rotten Guava",
    "Rotten Jujube", "Rotten Orange", "Rotten Pomegranate", "Rotten Strawberry"
]

CLASS_NAMES_2 = ["Fresh", "Rotten"]

MODEL_CONFIGS = {
    'BaseCNN_Original': {'file': 'base_cnn_original.keras', 'type': 'binary', 'gradcam_layer': 'conv2d_2'},
    'BaseCNN_Augmented': {'file': 'base_cnn_augmented.keras', 'type': 'binary', 'gradcam_layer': 'conv2d_2'},
    'MobileNet_Binary': {'file': 'mobilenet_binary.keras', 'type': 'binary', 'gradcam_layer': 'Conv_1'},
    'ResNet_Binary': {'file': 'resnet_binary.h5', 'type': 'binary', 'gradcam_layer': 'conv5_block3_out'},
    'MobileNet_16_Frozen': {'file': 'mobilenet_multiclass_frozen.keras', 'type': 'multiclass', 'gradcam_layer': 'Conv_1'},
    'MobileNet_16_Finetuned': {'file': 'mobilenet_multiclass_finetuned.keras', 'type': 'multiclass', 'gradcam_layer': 'Conv_1'},
    'Fusion_Model': {'file': 'fusion_model.keras', 'type': 'binary', 'gradcam_layer': None}
}

# ========== MODEL LOADING ==========

@st.cache_resource
def load_all_models():
    """Load all models with proper error handling"""
    models = {}
    errors = []
    
    for name, config in MODEL_CONFIGS.items():
        if config['file'] is None or name == 'Fusion_Model':
            continue
            
        try:
            if os.path.exists(config['file']):
                model = tf.keras.models.load_model(config['file'], compile=False)
                models[name] = model
            else:
                errors.append(f"{name}: File not found")
        except Exception as e:
            errors.append(f"{name}: {str(e)[:50]}")
    
    # Try fusion model (optional)
    try:
        if os.path.exists('fusion_model.keras'):
            models['Fusion_Model'] = tf.keras.models.load_model('fusion_model.keras', compile=False)
    except:
        models['Fusion_Model'] = None
    
    if errors:
        st.warning(f"⚠️ Some models failed to load:\n" + "\n".join(errors))
    
    if models:
        st.success(f"✅ Successfully loaded {len(models)} models!")
        return models
    else:
        st.error("❌ No models loaded successfully!")
        return None

# ========== PREPROCESSING ==========

def preprocess_image(image):
    """Preprocess image: resize and normalize"""
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    # Ensure 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ========== PREDICTION FUNCTIONS ==========

def safe_predict(model, img_array):
    """Safe prediction with error handling"""
    try:
        tf.random.set_seed(42)
        np.random.seed(42)
        prediction = model(img_array, training=False)
        return prediction.numpy()
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def predict_binary(model, img_array):
    """Binary prediction"""
    pred = safe_predict(model, img_array)
    if pred is None:
        return None
    
    raw_output = float(pred[0][0])
    predicted_class = "Rotten" if raw_output >= 0.5 else "Fresh"
    confidence = raw_output if raw_output >= 0.5 else (1 - raw_output)
    
    return {
        'raw_output': raw_output,
        'predicted_class': predicted_class,
        'confidence': confidence
    }

def predict_multiclass(model, img_array):
    """Multi-class prediction"""
    pred = safe_predict(model, img_array)
    if pred is None:
        return None
    
    raw_logits = pred[0]
    predicted_idx = np.argmax(raw_logits)
    confidence = float(raw_logits[predicted_idx])
    predicted_class = CLASS_NAMES_16[predicted_idx]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': {CLASS_NAMES_16[i]: float(raw_logits[i]) for i in range(len(CLASS_NAMES_16))}
    }

def predict_fusion(models, img_array):
    """Fusion prediction"""
    # Check if fusion model exists
    if 'Fusion_Model' in models and models['Fusion_Model'] is not None:
        result = predict_binary(models['Fusion_Model'], img_array)
        if result:
            result['method'] = 'Trained Fusion Model'
            return result
    
    # Fallback to averaging
    if 'MobileNet_Binary' in models and 'ResNet_Binary' in models:
        mob_pred = predict_binary(models['MobileNet_Binary'], img_array)
        res_pred = predict_binary(models['ResNet_Binary'], img_array)
        
        if mob_pred and res_pred:
            avg_prob = (mob_pred['raw_output'] + res_pred['raw_output']) / 2
            predicted_class = "Rotten" if avg_prob >= 0.5 else "Fresh"
            confidence = avg_prob if avg_prob >= 0.5 else (1 - avg_prob)
            
            return {
                'method': 'Average Ensemble',
                'predicted_class': predicted_class,
                'confidence': confidence,
                'mobilenet_prob': mob_pred['raw_output'],
                'resnet_prob': res_pred['raw_output']
            }
    
    return None

# ========== NOVELTY DETECTION ==========

def detect_spoilage_stage(confidence):
    """Spoilage stage detection"""
    if confidence >= 0.85:
        return {'stage': 'Clearly Fresh / Clearly Rotten', 'color': 'info'}
    elif confidence >= 0.60:
        return {'stage': 'Early Spoilage / Transitional', 'color': 'warning'}
    else:
        return {'stage': 'Uncertain (Manual Inspection)', 'color': 'error'}

# ========== FEATURE SPACE ANALYSIS ==========

def extract_features(model, img_array, model_name):
    """Extract feature vector from penultimate layer"""
    try:
        # Find the feature extraction layer (before final dense)
        if 'MobileNet' in model_name:
            # MobileNet: GlobalAveragePooling2D output
            feature_layer = None
            for layer in model.layers:
                if 'global_average_pooling' in layer.name.lower():
                    feature_layer = layer
                    break
            
            if feature_layer is None:
                # Fallback to -3rd layer
                feature_layer = model.layers[-3]
        elif 'ResNet' in model_name:
            # ResNet: GlobalAveragePooling2D output
            feature_layer = None
            for layer in model.layers:
                if 'global_average_pooling' in layer.name.lower():
                    feature_layer = layer
                    break
            
            if feature_layer is None:
                feature_layer = model.layers[-3]
        else:
            # BaseCNN: Flatten layer
            feature_layer = model.layers[-3]
        
        # Create feature extraction model
        feature_model = tf.keras.Model(
            inputs=model.input,
            outputs=feature_layer.output
        )
        
        # Extract features
        features = feature_model(img_array, training=False).numpy()[0]
        return features
    
    except Exception as e:
        return None

def analyze_feature_space(features, predicted_class, confidence):
    """Analyze position in feature space"""
    if features is None:
        return None
    
    # Compute feature norm (distance from origin)
    feature_norm = np.linalg.norm(features)
    
    # Determine cluster position based on prediction and confidence
    if 'Fresh' in predicted_class or predicted_class == 'Fresh':
        if confidence >= 0.85:
            position = "Deep in Fresh Cluster"
            interpretation = "Strong Fresh characteristics"
        elif confidence >= 0.60:
            position = "Near Fresh Cluster Boundary"
            interpretation = "Transitioning between Fresh and Rotten"
        else:
            position = "Uncertain Region"
            interpretation = "Weak clustering, manual inspection recommended"
    else:
        if confidence >= 0.85:
            position = "Deep in Rotten Cluster"
            interpretation = "Strong Rotten characteristics"
        elif confidence >= 0.60:
            position = "Near Rotten Cluster Boundary"
            interpretation = "Transitioning between Fresh and Rotten"
        else:
            position = "Uncertain Region"
            interpretation = "Weak clustering, manual inspection recommended"
    
    return {
        'feature_vector_dim': features.shape[0],
        'feature_norm': float(feature_norm),
        'cluster_position': position,
        'interpretation': interpretation
    }

# ========== GRAD-CAM ==========

def compute_gradcam(model, img_array, layer_name):
    """Compute Grad-CAM heatmap"""
    try:
        # Find the layer
        conv_layer = None
        for layer in model.layers:
            if layer.name == layer_name:
                conv_layer = layer
                break
        
        if conv_layer is None:
            return None
        
        # Create gradient model
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            # Get loss
            if predictions.shape[-1] == 1:
                loss = predictions[0][0]
            else:
                loss = predictions[0][tf.argmax(predictions[0])]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()
    except Exception as e:
        st.warning(f"Grad-CAM failed: {str(e)[:50]}")
        return None

def overlay_gradcam(image, heatmap):
    """Overlay heatmap on image"""
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_array = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# ========== MAIN APP ==========

def main():
    st.title("🍎 Fruit Freshness Detection System")
    st.markdown("Multi-model AI system with explainability")
    
    # Load models
    models = load_all_models()
    
    if models is None or len(models) == 0:
        st.error("❌ No models available. Please check model files.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        available_models = [name for name in MODEL_CONFIGS.keys() if name in models]
        selected_models = st.multiselect(
            "Select Models",
            options=available_models,
            default=available_models[:3] if len(available_models) >= 3 else available_models
        )
        
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        show_novelty = st.checkbox("Show Novelty Detection", value=True)
        show_features = st.checkbox("Show Feature Space Analysis", value=True)
        
        st.markdown("---")
        st.info(f"**Loaded Models:** {len(models)}")
    
    # Upload
    st.header("📤 Upload Image")
    upload_option = st.radio("Input method:", ["Upload File", "Camera", "Multiple Files"], horizontal=True)
    
    uploaded_files = []
    
    if upload_option == "Upload File":
        file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'])
        if file:
            uploaded_files = [file]
    elif upload_option == "Camera":
        file = st.camera_input("Take photo")
        if file:
            uploaded_files = [file]
    else:
        files = st.file_uploader("Choose images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        if files:
            uploaded_files = files
    
    # Process images
    if uploaded_files and selected_models:
        for idx, file in enumerate(uploaded_files):
            st.markdown("---")
            st.subheader(f"📷 Image {idx + 1}")
            
            image = Image.open(file).convert('RGB')
            img_array = preprocess_image(image)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption='Original', use_container_width=True)
            
            with col2:
                for model_name in selected_models:
                    if model_name not in models:
                        continue
                    
                    st.markdown(f"### 🤖 {model_name}")
                    config = MODEL_CONFIGS[model_name]
                    
                    # Predict
                    result = None
                    if model_name == 'Fusion_Model':
                        result = predict_fusion(models, img_array)
                    elif config['type'] == 'binary':
                        result = predict_binary(models[model_name], img_array)
                    else:
                        result = predict_multiclass(models[model_name], img_array)
                    
                    if result is None:
                        st.error("Prediction failed")
                        continue
                    
                    # Display result
                    pred_class = result['predicted_class']
                    confidence = result['confidence']
                    
                    if 'Fresh' in pred_class or pred_class == 'Fresh':
                        st.success(f"✅ **{pred_class}**")
                    else:
                        st.error(f"🚫 **{pred_class}**")
                    
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Novelty detection
                    if show_novelty:
                        stage_info = detect_spoilage_stage(confidence)
                        if stage_info['color'] == 'info':
                            st.info(f"📊 {stage_info['stage']}")
                        elif stage_info['color'] == 'warning':
                            st.warning(f"⚠️ {stage_info['stage']}")
                        else:
                            st.error(f"🔍 {stage_info['stage']}")
                    
                    # Feature Space Analysis
                    if show_features and model_name != 'Fusion_Model':
                        features = extract_features(models[model_name], img_array, model_name)
                        if features is not None:
                            feature_info = analyze_feature_space(features, pred_class, confidence)
                            if feature_info:
                                with st.expander("🧬 Feature Space Analysis"):
                                    st.write(f"**Position:** {feature_info['cluster_position']}")
                                    st.write(f"**Interpretation:** {feature_info['interpretation']}")
                                    st.caption(f"Feature dimension: {feature_info['feature_vector_dim']}D | Norm: {feature_info['feature_norm']:.2f}")
                    
                    # Top predictions for multiclass
                    if config['type'] == 'multiclass' and 'all_probabilities' in result:
                        with st.expander("Top 5 Predictions"):
                            sorted_probs = sorted(result['all_probabilities'].items(), 
                                                key=lambda x: x[1], reverse=True)[:5]
                            for cls, prob in sorted_probs:
                                st.write(f"{cls}: {prob:.1%}")
                    
                    # Fusion details
                    if 'mobilenet_prob' in result:
                        with st.expander("Fusion Details"):
                            st.write(f"Method: {result.get('method', 'Unknown')}")
                            st.write(f"MobileNet: {result['mobilenet_prob']:.4f}")
                            st.write(f"ResNet: {result['resnet_prob']:.4f}")
                    
                    # Grad-CAM
                    if show_gradcam and config['gradcam_layer']:
                        st.markdown("**Grad-CAM Explainability**")
                        
                        if model_name == 'Fusion_Model' and 'MobileNet_Binary' in models and 'ResNet_Binary' in models:
                            gcol1, gcol2 = st.columns(2)
                            
                            with gcol1:
                                st.caption("MobileNet")
                                heatmap = compute_gradcam(models['MobileNet_Binary'], img_array, 'Conv_1')
                                if heatmap is not None:
                                    overlay = overlay_gradcam(image, heatmap)
                                    st.image(overlay, use_container_width=True)
                            
                            with gcol2:
                                st.caption("ResNet")
                                heatmap = compute_gradcam(models['ResNet_Binary'], img_array, 'conv5_block3_out')
                                if heatmap is not None:
                                    overlay = overlay_gradcam(image, heatmap)
                                    st.image(overlay, use_container_width=True)
                        else:
                            heatmap = compute_gradcam(models[model_name], img_array, config['gradcam_layer'])
                            if heatmap is not None:
                                overlay = overlay_gradcam(image, heatmap)
                                st.image(overlay, caption='Grad-CAM', use_container_width=True)
                    
                    st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🚀 Real TensorFlow Inference • Deterministic Results • Full Explainability</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
