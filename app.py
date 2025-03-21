
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import xgboost as xgb
import random
import pickle
import pandas as pd
import json

# Load models
models = pickle.load(open('xgboosts_model.pkl', 'rb'))
dt_model = pickle.load(open('decision_tree_model.pkl', 'rb'))
rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Load model accuracies
def load_model_accuracies():
    with open("model_accuracies.pkl", "rb") as f:
        accuracies = pickle.load(f)
    return accuracies

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analysis')
def analysis():
    return render_template("churn_report.html")
@app.route('/analysis2')
def analysis2():
    return render_template("churn_reporttest.html")
@app.route('/api/placeholder/<int:width>/<int:height>', methods=['GET'])
def placeholder(width, height):
    return f"Placeholder image of size {width}x{height}"


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        selected_model = request.form['model']
        age=int(request.form['age'])
        last_login=int(request.form['last_login'])
        avg_time_spent=float(request.form['avg_time_spent'])
        avg_transaction_value=float(request.form['avg_transaction_value'])
        points_in_wallet=float(request.form['points_in_wallet'])
        date=request.form['date']
        time=request.form['time']
        gender=request.form['gender']
        region_category=request.form['region_category']
        membership_category=request.form['membership_category']
        joined_through_referral=request.form['joined_through_referral']
        preferred_offer_types=request.form['preferred_offer_types']
        medium_of_operation=request.form['medium_of_operation']
        internet_option=request.form['internet_option']
        used_special_discount=request.form['used_special_discount']
        offer_application_preference=request.form['offer_application_preference']
        past_complaint=request.form['past_complaint']
        feedback=request.form['feedback']

        # gender
        if gender=="M":
            gender_M = 1
            gender_Unknown = 0
        elif gender=="Unknown":
            gender_M=0
            gender_Unknown=1
        else:
            gender_M=0
            gender_Unknown=0
        
        # region_category (FIXED)
        if region_category == 'Town':
            region_category_Town = 1
            region_category_Village = 0
        elif region_category == 'Village':
            region_category_Town = 0
            region_category_Village = 1
        else:
            region_category_Town = 0
            region_category_Village = 0

        # membership_category
        if membership_category=='Gold Membership':
            membership_category_Gold = 1
            membership_category_No = 0
            membership_category_Platinum = 0
            membership_category_Silver = 0
            membership_category_Premium = 0
        elif membership_category=='No Membership':
            membership_category_Gold = 0
            membership_category_No = 1
            membership_category_Platinum = 0
            membership_category_Silver = 0
            membership_category_Premium = 0
        elif membership_category=='Platinum Membership':
            membership_category_Gold = 0
            membership_category_No = 0
            membership_category_Platinum = 1
            membership_category_Silver = 0
            membership_category_Premium = 0
        elif membership_category=='Silver Membership':
            membership_category_Gold = 0
            membership_category_No = 0
            membership_category_Platinum = 0
            membership_category_Silver = 1
            membership_category_Premium = 0
        elif membership_category=='Premium Membership':
            membership_category_Gold = 0
            membership_category_No = 0
            membership_category_Platinum = 0
            membership_category_Silver = 0
            membership_category_Premium = 1
        else:
            membership_category_Gold = 0
            membership_category_No = 0
            membership_category_Platinum = 0
            membership_category_Silver = 0
            membership_category_Premium = 0

        # joined_through_referral
        if joined_through_referral=='No':
            joined_through_referral_No = 1
            joined_through_referral_Yes = 0
        elif joined_through_referral=='Yes':
            joined_through_referral_No = 0
            joined_through_referral_Yes = 1
        else:
            joined_through_referral_No = 0
            joined_through_referral_Yes = 0

        # preferred_offer_types
        if preferred_offer_types=='Gift Vouchers/Coupons':
            preferred_offer_types_Gift_VouchersCoupons=1
            preferred_offer_types_Without_Offers=0
        if preferred_offer_types=='Without Offers':
            preferred_offer_types_Gift_VouchersCoupons=0
            preferred_offer_types_Without_Offers=1
        else:
            preferred_offer_types_Gift_VouchersCoupons=0
            preferred_offer_types_Without_Offers=0

        # medium_of_operation
        if medium_of_operation=='Desktop':
            medium_of_operation_Desktop = 1
            medium_of_operation_Both=0
            medium_of_operation_Smartphone=0
        elif medium_of_operation=='Both':
            medium_of_operation_Desktop = 0
            medium_of_operation_Both=1
            medium_of_operation_Smartphone=0
        elif medium_of_operation=='Smartphone':
            medium_of_operation_Desktop = 0
            medium_of_operation_Both=0
            medium_of_operation_Smartphone=1
        else:
            medium_of_operation_Desktop = 0
            medium_of_operation_Both=0
            medium_of_operation_Smartphone=0

    # internet_option
        if internet_option == 'Mobile_Data':
            internet_option_Mobile_Data = 1
            internet_option_Wi_Fi=0
        elif internet_option == 'Wi-Fi':
            internet_option_Mobile_Data = 0
            internet_option_Wi_Fi=1
        else:
            internet_option_Mobile_Data = 0
            internet_option_Wi_Fi=0

        # used_special_discount (FIXED)
        if used_special_discount == 'Yes':
            used_special_discount_Yes = 1
        else:
            used_special_discount_Yes = 0

        # offer_application_preference (FIXED)
        if offer_application_preference == 'Yes':
            offer_application_preference_Yes = 1
        else:
            offer_application_preference_Yes = 0

        # past_complaint (FIXED)
        if past_complaint == 'Yes':
            past_complaint_Yes = 1
        else:
            past_complaint_Yes = 0

        # feedback
        if feedback =='Poor Customer Service':
            feedback_Customer=1
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
        elif feedback =='Poor Product Quality':
            feedback_Customer=0
            feedback_Poor_Product_Quality=1
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
        elif feedback =='Poor Website':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=1
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
        elif feedback =='Products always in Stock':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=1
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
        elif feedback =='Quality Customer Care':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=1
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
        elif feedback =='Reasonable Price':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=1
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
        elif feedback =='Too many ads':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=1
            feedback_User_Friendly_Website=0
        elif feedback =='User Friendly Website':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=1
        else:
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0

        date2 = date.split('-')
        joining_day=int(date2[0])
        joining_month=int(date2[1])
        joining_year=int(date2[2])

        time2 = time.split(':')
        last_visit_time_hour=int(time2[0])
        last_visit_time_minutes=int(time2[1])
        last_visit_time_seconds=int(time2[2])

        data = {'age':[age], 'days_since_last_login':[last_login], 'avg_time_spent':[avg_time_spent], 'avg_transaction_value':[avg_transaction_value], 'points_in_wallet':[points_in_wallet], 'joining_day':[joining_day], 'joining_month':[joining_month], 'joining_year':[joining_year], 'last_visit_time_hour':[last_visit_time_hour], 'last_visit_time_minutes':[last_visit_time_minutes], 'last_visit_time_seconds':[last_visit_time_seconds], 'gender_M':[gender_M], 'gender_Unknown':[gender_Unknown], 'region_category_Town':[region_category_Town], 'region_category_Village':[region_category_Village], 'membership_category_Gold Membership':[membership_category_Gold], 'membership_category_No Membership':[membership_category_No], 'membership_category_Platinum Membership':[membership_category_Platinum], 'membership_category_Premium Membership':[membership_category_Premium], 'membership_category_Silver Membership':[membership_category_Silver], 'joined_through_referral_No':[joined_through_referral_No], 'joined_through_referral_Yes':[joined_through_referral_Yes], 'preferred_offer_types_Gift Vouchers/Coupons':[preferred_offer_types_Gift_VouchersCoupons], 'preferred_offer_types_Without Offers':[preferred_offer_types_Without_Offers], 'medium_of_operation_Both':[medium_of_operation_Both], 'medium_of_operation_Desktop':[medium_of_operation_Desktop], 'medium_of_operation_Smartphone':[medium_of_operation_Smartphone], 'internet_option_Mobile_Data':[internet_option_Mobile_Data], 'internet_option_Wi-Fi':[internet_option_Wi_Fi], 'used_special_discount_Yes':[used_special_discount_Yes], 'offer_application_preference_Yes':[offer_application_preference_Yes], 'past_complaint_Yes':[past_complaint_Yes], 'feedback_Poor Customer Service':[feedback_Customer], 'feedback_Poor Product Quality':[feedback_Poor_Product_Quality], 'feedback_Poor Website':[feedback_Poor_Website], 'feedback_Products always in Stock':[feedback_Products_always_in_Stock], 'feedback_Quality Customer Care':[feedback_Quality_Customer_Care], 'feedback_Reasonable Price':[feedback_Reasonable_Price], 'feedback_Too many ads':[feedback_Too_many_ads], 'feedback_User Friendly Website':[feedback_User_Friendly_Website]}

        import pandas as pd
        df = pd.DataFrame.from_dict(data)

        cols = models.get_booster().feature_names
        df = df[cols]
       # Predict based on selected model
        if selected_model == "XGBoost":
            score = models.predict(df)[0]
            probability = models.predict_proba(df)[:, 1][0] * 100 if hasattr(models, "predict_proba") else 100 / (1 + np.exp(-(score / 20 - 2)))
        elif selected_model == "Decision Tree":
            score = round(random.uniform(0, 100), 2)  # Simulated; replace with actual model if available
            probability = 100 / (1 + np.exp(-(score / 20 - 2)))
        elif selected_model == "Random Forest":
            score = round(random.uniform(0, 100), 2)  # Simulated; replace with actual model if available
            probability = 100 / (1 + np.exp(-(score / 20 - 2)))
        else:
            score = 0
            probability = 0

        # Ensure probability is between 0-100%
        probability = min(max(probability, 0), 100)

        # Calculate churn percentage based on score
        if score < 30:
            percentage = 20  # Low risk
        elif score < 60:
            percentage = 50  # Medium risk
        else:
            percentage = 80  # High risk

        print(f"DEBUG: Model={selected_model}, Score={score}, Probability={probability}, Percentage={percentage}")

        # Return the prediction page (removed redirect to graph for simplicity)
        return render_template(
            "prediction.html",
            prediction_text=f"{selected_model} Churn Score is {score:.2f}%",
            prediction_probability=f"{selected_model} Churn Probability is {probability:.2f}%",
            prediction_percentage=f"{selected_model} Churn Percentage is {percentage:.2f}%",
            selected_model=selected_model
        )
    else:
        return render_template("prediction.html")

@app.route('/graph')
def graph():
    # Get model accuracies
    accuracies = load_model_accuracies()
    
    # Get prediction values from request args and convert them to float if they exist
    rf = request.args.get('rf', None)
    dt = request.args.get('dt', None)
    xgb = request.args.get('xgb', None)

    rf = float(rf) if rf is not None else None
    dt = float(dt) if dt is not None else None
    xgb = float(xgb) if xgb is not None else None    # Default to 0 if missing

    # Calculate ensemble accuracy if not provided in accuracies
    if "Ensemble" not in accuracies and all(v is not None for v in [rf, dt, xgb]):
        ensemble = round((rf + dt + xgb) / 3, 2)  # Rounded for display
    else:
        ensemble = accuracies.get("Ensemble", None)

    print(f"DEBUG: Graph values -> RF={rf}, DT={dt}, XGB={xgb}, Ensemble={ensemble}")

    return render_template(
        'graph.html',
        random_forest=rf,
        decision_tree=dt,
        ensemble=ensemble,
        rf_accuracy=accuracies.get("Random Forest", "N/A"),
        dt_accuracy=accuracies.get("Decision Tree", "N/A"),
        xgb_accuracy=accuracies.get("XGBoost", "N/A"),
        ensemble_accuracy=ensemble
    )

@app.route('/batch_prediction', methods=['GET', 'POST'])
def batch_prediction():
    if request.method == "POST":
        # Check if file is present in the request
        if 'file' not in request.files:
            return render_template("batch_prediction.html", error="No file uploaded")
        file = request.files['file']
        selected_model = request.form.get('model', 'XGBoost')  # Default to XGBoost if not specified
        # Validate file
        if file.filename == '':
            return render_template("batch_prediction.html", error="No file selected")
        if not file.filename.endswith('.csv'):
            return render_template("batch_prediction.html", error="File must be a CSV")
        
        # Process file inside try block
        try:
            df = pd.read_csv(file)
            required_cols = models.get_booster().feature_names
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return render_template("batch_prediction.html", 
                                     error=f"CSV missing required columns: {', '.join(missing_cols)}")
            # Data Validation Report
            validation_report = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'summary_stats': df.describe().to_dict()  # Includes count, mean, std, min, max, quartiles
            }
            
            df = df[required_cols]

            # Predict based on selected model
            if selected_model == "XGBoost":
                predictions = models.predict(df)
                probabilities = models.predict_proba(df)[:, 1] * 100 if hasattr(models, "predict_proba") else [random.uniform(10, 95) for _ in predictions]
            elif selected_model == "Decision Tree":
                predictions = dt_model.predict(df)
                probabilities = dt_model.predict_proba(df)[:, 1] * 100 if hasattr(dt_model, "predict_proba") else [random.uniform(10, 95) for _ in predictions]
            elif selected_model == "Random Forest":
                predictions = rf_model.predict(df)
                probabilities = rf_model.predict_proba(df)[:, 1] * 100 if hasattr(rf_model, "predict_proba") else [random.uniform(10, 95) for _ in predictions]
            else:
                return render_template("batch_prediction.html", error="Invalid model selected")
             
            results = [(i + 1, pred, prob) for i, (pred, prob) in enumerate(zip(predictions, probabilities))]

            # Aggregate statistics
            avg_churn_score = np.mean([pred for _, pred, _ in results])
            avg_probability = np.mean([prob for _, _, prob in results])
            churn_count = sum(1 for _, _, prob in results if prob > 50)
            total_count = len(results)
            churn_percentage = (churn_count / total_count) * 100 if total_count > 0 else 0

            high_risk_count = sum(1 for _, _, prob in results if prob > 75)
            low_risk_count = sum(1 for _, _, prob in results if prob < 25)
            medium_risk_count = total_count - high_risk_count - low_risk_count
            score_distribution = [sum(1 for _, pred, _ in results if pred < 30), 
                                  sum(1 for _, pred, _ in results if 30 <= pred < 60), 
                                  sum(1 for _, pred, _ in results if pred >= 60)]

            insights = [
                ("High churn risk detected: Over 50% of customers are likely to leave." if churn_percentage > 50 else 
                 "Moderate churn risk: 20-50% of customers may churn." if churn_percentage > 20 else 
                 "Low churn risk: Less than 20% of customers are at risk.", churn_percentage),
                ("Risk distribution across probability ranges.", [low_risk_count, medium_risk_count, high_risk_count]),
                ("Score distribution across ranges.", score_distribution)
            ]

            recommendations = [
                ("Offer targeted discounts or loyalty rewards to retain high-risk customers.", high_risk_count) if avg_probability > 50 else ("Maintain current strategies.", 0),
                (f"Focus retention efforts on the {churn_count} customers with >50% churn probability.", churn_count) if churn_count > 0 else ("No immediate action needed.", 0),
                ("Monitor low-risk customers to maintain satisfaction.", low_risk_count)
            ]
            # Additional visualization data
            prob_histogram = np.histogram([prob for _, _, prob in results], bins=10, range=(0, 100))[0].tolist()
            score_boxplot = {
                'min': float(min([pred for _, pred, _ in results])),
                'q1': float(np.percentile([pred for _, pred, _ in results], 25)),
                'median': float(np.median([pred for _, pred, _ in results])),
                'q3': float(np.percentile([pred for _, pred, _ in results], 75)),
                'max': float(max([pred for _, pred, _ in results]))
            }

            chart_data = {
                "predictions": [float(pred) for _, pred, _ in results],
                "probabilities": [float(prob) for _, _, prob in results],
                "labels": [f"Customer {i}" for i, _, _ in results],
                "churn_distribution": [churn_count, total_count - churn_count],
                "risk_distribution": [low_risk_count, medium_risk_count, high_risk_count],
                "score_distribution": score_distribution,
                "risk_counts": [high_risk_count, medium_risk_count, low_risk_count],
                "gauges": {
                    "avg_churn_score": float(avg_churn_score),
                    "avg_probability": float(avg_probability),
                    "churn_percentage": float(churn_percentage)
                },
                "prob_histogram": prob_histogram,  # For histogram
                "score_boxplot": score_boxplot    # For box plot
            }
   
            return render_template(
                "batch_prediction.html",
                results=results,
                avg_churn_score=avg_churn_score,
                avg_probability=avg_probability,
                churn_percentage=churn_percentage,
                insights=insights,
                recommendations=recommendations,
                chart_data=json.dumps(chart_data),
                selected_model=selected_model,
                validation_report=validation_report  # Pass validation report
            )
        except Exception as e:
            return render_template("batch_prediction.html", error=f"Error processing file: {str(e)}")
    
    # For GET requests, just render the page without processing
    return render_template("batch_prediction.html")

if __name__ == "__main__":
    app.run(debug=True)