from flask import Flask, render_template, request
import pandas as pd
import joblib
import io
import base64
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load Model and Features
try:
    payload = joblib.load('best_student_model.pkl')
    model = payload['model']
    features = payload['features']
except Exception as e:
    print(f"Error loading model: {e}")
    model, features = None, []

@app.route('/')
def home():
    return render_template('index.html', results=None, graph_url=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return "Model not loaded. Check server logs."

        # 1. Capture Form Data (Supporting multiple students via list)
        # We assume the form sends lists for each field
        genders = request.form.getlist('gender')
        ethnicities = request.form.getlist('race/ethnicity')
        parents = request.form.getlist('parental level of education')
        lunches = request.form.getlist('lunch')
        preps = request.form.getlist('test preparation course')
        math_scores = request.form.getlist('math score')
        reading_scores = request.form.getlist('reading score')
        writing_scores = request.form.getlist('writing score') # Added Subject

        student_data = []
        for i in range(len(genders)):
            student_data.append({
                'gender': genders[i],
                'race/ethnicity': ethnicities[i],
                'parental level of education': parents[i],
                'lunch': lunches[i],
                'test preparation course': preps[i],
                'math score': float(math_scores[i]),
                'reading score': float(reading_scores[i]),
                'writing score': float(writing_scores[i])
            })

        # 2. Process for AI
        df_input = pd.DataFrame(student_data)
        df_encoded = pd.get_dummies(df_input)
        final_input = df_encoded.reindex(columns=features, fill_value=0)
        
        predictions = model.predict(final_input)
        
        # 3. Format Results
        display_results = []
        for i, pred in enumerate(predictions):
            score = round(pred * 100, 2) if pred <= 1.0 else round(pred, 2)
            display_results.append({
                'id': i + 1,
                'math': math_scores[i],
                'reading': reading_scores[i],
                'writing': writing_scores[i],
                'prediction': score
            })

        # 4. Comparative Visualization (Averages)
        avg_math = df_input['math score'].mean()
        avg_reading = df_input['reading score'].mean()
        avg_writing = df_input['writing score'].mean()
        avg_pred = sum([r['prediction'] for r in display_results]) / len(display_results)

        plt.figure(figsize=(8, 5))
        subjects = ['Math', 'Reading', 'Writing', 'AI Avg']
        vals = [avg_math, avg_reading, avg_writing, avg_pred]
        colors = ['#3498db', '#5dade2', '#aed6f1', '#2c3e50']
        
        bars = plt.bar(subjects, vals, color=colors)
        plt.title('Group Performance vs. AI Prediction')
        plt.ylabel('Score %')
        plt.ylim(0, 110)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{round(yval,1)}%', ha='center')

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template('index.html', results=display_results, graph_url=graph_url)

    except Exception as e:
        return f"Prediction Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)