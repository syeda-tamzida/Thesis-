import cv2
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import gradio as gr
import os
from PIL import Image
import io
import base64

# Step 4: Define the model training function (updated for new dataset)
def train_corrosion_model(data_path='data/data.csv'):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    
    df = pd.read_csv(data_path)
    # Check required columns
    required_cols = ['temperature_c', 'humidity_%', 'half-cell_potential_v', 'surface_resistivity', 'surface_roughness'] #change kw
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Feature engineering
    df['log_resistivity'] = np.log(df['surface_resistivity'])
    df['roughness_resistivity_ratio'] = df['surface_roughness'] / df['surface_resistivity']
    df['temp_humidity_interaction'] = df['temperature_c'] * df['humidity_%']
    
    # Features and target
    features = [
        'surface_resistivity',
        'surface_roughness',
        'temperature_c',
        'humidity_%',
        'log_resistivity',
        'roughness_resistivity_ratio',
        'temp_humidity_interaction'
    ]
    target = 'half-cell_potential_v'
    
    X = df[features]
    y = df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    joblib.dump(model, 'corrosion_model.pkl')
    joblib.dump(scaler, 'corrosion_scaler.pkl')

    return model, scaler

    
# Step 5: Image processing function (same as before)
def extract_and_visualize_roughness(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

        roughness = np.std(gradient_magnitude)



        return roughness

    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return None

# Step 6: Enhanced CorrosionPredictor with environmental factors

class CorrosionPredictor:
    def __init__(self, model_path='corrosion_model.pkl', scaler_path='corrosion_scaler.pkl', train_new=True):
        if train_new:
            self.model, self.scaler = train_corrosion_model()
        else:
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"Error loading model/scaler: {e}")
                self.model = None
                self.scaler = None

    def predict_from_image(self, image_path, resistivity, temperature=25.0, humidity=50.0):
        if not self.model or not self.scaler:
            print("Model or scaler not loaded properly.")
            return None

        roughness = extract_and_visualize_roughness(image_path)
        if roughness is None:
            return None

        input_features = {
            'surface_resistivity': resistivity,
            'surface_roughness': roughness,
            'temperature_c': temperature,
            'humidity_%': humidity,
            'log_resistivity': np.log(resistivity),
            'roughness_resistivity_ratio': roughness / resistivity,
            'temp_humidity_interaction': temperature * humidity
        }

        feature_order = [
            'surface_resistivity',
            'surface_roughness',
            'temperature_c',
            'humidity_%',
            'log_resistivity',
            'roughness_resistivity_ratio',
            'temp_humidity_interaction'
        ]

        input_df = pd.DataFrame([input_features])[feature_order]
        scaled_input = self.scaler.transform(input_df)
        hcp = self.model.predict(scaled_input)[0]

        if hcp > -0.2:
            prob_class = "Low (<10%)"
            prob_desc = "Low corrosion activity"
        elif -0.35 <= hcp <= -0.2:
            prob_class = "Moderate (10-90%)"
            prob_desc = "Moderate corrosion risk"
        else:
            prob_class = "High (>90%)"
            prob_desc = "High corrosion activity"

        results = {
            "Image_Path": image_path,
            "Input_Resistivity": resistivity,
            "Temperature_C": temperature,
            "Humidity_%": humidity,
            "Measured_Roughness": round(roughness, 4),
            "Predicted_HCP_V": round(hcp, 5),
            "Corrosion_Probability": prob_class,
            "Risk_Description": prob_desc,
            "ASTM_Interpretation": self.get_astm_interpretation(hcp)
        }

        self.display_results(image_path, results)
        return results

    def get_astm_interpretation(self, hcp):
        if hcp > -0.2:
            return ">90% probability no corrosion"
        elif -0.35 <= hcp <= -0.2:
            return "Corrosion activity uncertain"
        else:
            return ">90% probability of corrosion"

    def display_results(self, image_path, results):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annotation_text = (
            f"Predicted HCP: {results['Predicted_HCP_V']} V CSE\n"
            f"Resistivity: {results['Input_Resistivity']} kΩ\n"
            f"Temperature: {results['Temperature_C']}°C\n"
            f"Humidity: {results['Humidity_%']}%\n"
            f"Surface Roughness: {results['Measured_Roughness']}\n"
            f"Corrosion Probability: {results['Corrosion_Probability']}\n"
            f"ASTM: {results['ASTM_Interpretation']}"
        )




# Step 7: Enhanced Gradio interface with environmental factors
predictor = CorrosionPredictor(train_new=True)  # Set False if you want to load model from file

def corrosion_chatbot(image, resistivity, temperature, humidity):
    if image is None:
        return "Please upload an image.", None, ""

    try:
        resistivity_val = float(resistivity)
        temp_val = float(temperature)
        hum_val = float(humidity)
    except:
        return "Please enter valid numeric values for all parameters.", None, ""

    # Save uploaded PIL image to disk
    image_path = "/tmp/uploaded_concrete_image.png"
    image.save(image_path)

    results = predictor.predict_from_image(image_path, resistivity_val, temp_val, hum_val)
    if not results:
        return "Prediction failed. Check your inputs or model.", None, ""

    # Get basic results
    basic_results = (
        f"Predicted HCP: {results['Predicted_HCP_V']} V\n"
        f"Resistivity: {results['Input_Resistivity']} kΩ\n"
        f"Temperature: {results['Temperature_C']}°C\n"
        f"Humidity: {results['Humidity_%']}%\n"
        f"Surface Roughness: {results['Measured_Roughness']}\n"
        f"Corrosion Probability: {results['Corrosion_Probability']}\n"
        f"ASTM Interpretation: {results['ASTM_Interpretation']}"
    )

    return basic_results, image

# Create Gradio interface with 3 output components
interface = gr.Interface(
    fn=corrosion_chatbot,
    inputs=[
        gr.Image(type="pil", label="Concrete Surface Image"),
        gr.Textbox(label="Concrete Resistivity (kΩ)"),
        gr.Slider(minimum=-10, maximum=50, value=25, label="Temperature (°C)"),
        gr.Slider(minimum=0, maximum=100, value=50, label="Humidity (%)")
    ],
    outputs=[
        gr.Textbox(label="Basic Results"),
        gr.Image(label="Uploaded Image")
    ],
    title="Advanced Concrete Corrosion Prediction with Environmental Factors",

    examples=[
        ["image/one.jpg", "51.9", 30, 48],
        ["image/two.jpg", "21.2", 32, 80]
    ]
)

interface.launch()