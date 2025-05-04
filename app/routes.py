import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the pre-trained model
model_path = 'model/xgb_model.pkl'
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form
        HadAngina = 1 if request.form['HadAngina'] == 'Yes' else 0
        AgeCategory = int(request.form['AgeCategory'])
        LastCheckupTime = int(request.form['LastCheckupTime'])
        PhysicalActivities = 1 if request.form['PhysicalActivities'] == 'Yes' else 0
        PhysicalHealthDays = int(request.form['PhysicalHealthDays'])
        MentalHealthDays = int(request.form['MentalHealthDays'])
        SleepHours = int(request.form['SleepHours'])
        BMI = float(request.form['BMI'])
        Sex = int(request.form['Sex'])
        GeneralHealth = int(request.form['GeneralHealth'])
        
        defaultValue = 0
        RemovedTeeth = defaultValue # 1 if request.form['RemovedTeeth'] == 'Yes' else 0
        HadStroke = defaultValue # 1 if request.form['HadStroke'] == 'Yes' else 0
        HadAsthma = defaultValue # 1 if request.form['HadAsthma'] == 'Yes' else 0
        HadSkinCancer = defaultValue # 1 if request.form['HadSkinCancer'] == 'Yes' else 0
        HadCOPD = defaultValue # 1 if request.form['HadCOPD'] == 'Yes' else 0
        HadDepressiveDisorder = defaultValue # 1 if request.form['HadDepressiveDisorder'] == 'Yes' else 0
        HadKidneyDisease = defaultValue # 1 if request.form['HadKidneyDisease'] == 'Yes' else 0
        HadArthritis = defaultValue # 1 if request.form['HadArthritis'] == 'Yes' else 0
        HadDiabetes = defaultValue # 1 if request.form['HadDiabetes'] == 'Yes' else 0
        DeafOrHardOfHearing = defaultValue # 1 if request.form['DeafOrHardOfHearing'] == 'Yes' else 0
        BlindOrVisionDifficulty = defaultValue # 1 if request.form['BlindOrVisionDifficulty'] == 'Yes' else 0
        DifficultyConcentrating = defaultValue # 1 if request.form['DifficultyConcentrating'] == 'Yes' else 0
        DifficultyWalking = defaultValue # 1 if request.form['DifficultyWalking'] == 'Yes' else 0
        DifficultyDressingBathing = defaultValue # 1 if request.form['DifficultyDressingBathing'] == 'Yes' else 0
        DifficultyErrands = defaultValue # 1 if request.form['DifficultyErrands'] == 'Yes' else 0
        SmokerStatus = defaultValue # 1 if request.form['SmokerStatus'] == 'Yes' else 0
        ECigaretteUsage = defaultValue # 1 if request.form['ECigaretteUsage'] == 'Yes' else 0
        ChestScan = defaultValue # 1 if request.form['ChestScan'] == 'Yes' else 0
        
        AlcoholDrinkers = defaultValue # 1 if request.form['AlcoholDrinkers'] == 'Yes' else 0
        HIVTesting = defaultValue # 1 if request.form['HIVTesting'] == 'Yes' else 0
        FluVaxLast12 = defaultValue # 1 if request.form['FluVaxLast12'] == 'Yes' else 0
        PneumoVaxEver = defaultValue # 1 if request.form['PneumoVaxEver'] == 'Yes' else 0
        TetanusLast10Tdap = defaultValue # 1 if request.form['TetanusLast10Tdap'] == 'Yes' else 0
        HighRiskLastYear = defaultValue # 1 if request.form['HighRiskLastYear'] == 'Yes' else 0
        CovidPos = defaultValue # int(request.form['CovidPos'])

        # Combine all features into a numpy array
        features = np.array([[PhysicalHealthDays, MentalHealthDays, SleepHours, BMI, Sex, GeneralHealth,
                              LastCheckupTime, PhysicalActivities, RemovedTeeth, HadAngina, HadStroke, HadAsthma,
                              HadSkinCancer, HadCOPD, HadDepressiveDisorder, HadKidneyDisease, HadArthritis,
                              HadDiabetes, DeafOrHardOfHearing, BlindOrVisionDifficulty, DifficultyConcentrating,
                              DifficultyWalking, DifficultyDressingBathing, DifficultyErrands, SmokerStatus,
                              ECigaretteUsage, ChestScan, AgeCategory, AlcoholDrinkers, HIVTesting, FluVaxLast12,
                              PneumoVaxEver, TetanusLast10Tdap, HighRiskLastYear, CovidPos]])

        # Predict using the loaded model
        prediction = model.predict(features)[0]

        # Map prediction to readable text
        result = "No heart attack" if prediction == 0 else "Has heart attack"

        return render_template('index.html', prediction=result)
    except KeyError as e:
        return f"Missing or incorrect form field: {str(e)}", 400
    except Exception as e:
        return f"An error occurred: {str(e)}", 500
