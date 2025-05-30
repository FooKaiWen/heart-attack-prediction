# Heart Disease Prediction Web App

This project is a user-friendly web application that predicts the risk of heart disease based on user-provided health and lifestyle information. It leverages a machine learning model (XGBoost) trained on health survey data to provide personalized risk assessments and actionable recommendations.

---

## 🚀 Features

- **Interactive Survey:** Collects key health and lifestyle data from users.
- **BMI Auto-Calculation:** Calculates BMI from height and weight inputs.
- **Personalized Results:** Provides risk prediction, tailored recommendations, and motivational messages.
- **Premium Questions:** Optional extra questions for more detailed insights (can be unlocked).
- **Modern UI:** Responsive design using Bootstrap 5.


---

## ⚙️ How to Run Locally

1. **Clone the repository**
    ```bash
    git clone https://github.com/FooKaiWen/heart-attack-prediction
    cd heart-disease-prediction
    ```

2. **Create and activate a virtual environment (optional but recommended)**
    ```bash
    python -m venv venv
    venv\Scripts\activate   # On Windows
    # source venv/bin/activate   # On Mac/Linux
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Ensure the trained model file is present**
    - The file `model/xgb_model.pkl` should exist. If not, train the model or obtain it from the project owner.

5. **Run the Flask app**
    ```bash
    cd app
    python routes.py
    ```
    - The app will typically be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🌐 Deployment

To deploy on a production server (e.g., Heroku, Azure, AWS, etc.):

1. **Set up environment variables as needed (e.g., `FLASK_ENV=production`).**
2. **Use a production-ready WSGI server like Gunicorn:**
    ```bash
    gunicorn -w 4 routes:app
    ```
3. **Configure your platform to serve the app and static files.**

---

## 📄 License

This project is for educational purposes. Please contact the author for commercial use.

---

## 🙏 Acknowledgements

- [XGBoost](https://xgboost.ai/)
- [Flask](https://flask.palletsprojects.com/)
- [Bootstrap](https://getbootstrap.com/)
- Heart Disease UCI dataset and related health survey data

---

*Feel free to contribute or open issues for improvements!*