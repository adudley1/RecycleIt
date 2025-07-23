from flask import Flask, request, render_template, redirect, url_for
import os
import subprocess
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

NET = "/home/nvidia/jetson-inference/python/training/classification/models/recycle"
DATASET = "/home/nvidia/jetson-inference/python/training/classification/data/recycle"

recyclable = ["aluminum_soda_cans", "cardboard_packaging", "glass_beverage_bottles", "glass_cosmetic_containers", "glass_food_jars", "magazines", "newspaper", "office_paper", "paper_cups", "plastic_cup_lids", "plastic_detergent_bottles", "plastic_food_containers", "plastic_soda_bottles", "plastic_water_bottles", "steel_food_cans"]
conditionally_recyclable = ["plastic_shopping_bags", "plastic_trash_bags", "plastic_straws", "styrofoam_cups", "styrofoam_food_containers", "disposable_plastic_cutlery"]
compostable = ["coffee_grounds", "eggshells", "food_waste"]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = datetime.now().strftime("%Y%m%d-%H%M%S_") + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")

            command = [
                "imagenet.py",
                f"--model={NET}/resnet18.onnx",
                f"--labels={DATASET}/labels.txt",
                "--input_blob=input_0",
                "--output_blob=output_0",
                filepath,
                output_path
            ]

            try:
                output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)

                for line in output.splitlines():
                    if "class" in line.lower() and "%" in line:
                        result = line

                        if any(item in result for item in recyclable):
                            result += "\n✅ Recyclable"
                        elif any(item in result for item in conditionally_recyclable):
                            result += "\n✅ Conditionally Recyclable (requires special facilities)"
                        elif any(item in result for item in compostable):
                            result += "\n🍂 Compostable"
                        break
                else:
                    result = "✅ Image processed, but no prediction found."

            except subprocess.CalledProcessError as e:
                result = f"❌ Error:\n{e.output}"

            # Log the result
            if result:
                with open("classification_log.txt", "a") as log_file:
                    log_file.write(f"{datetime.now()} - {result}\n")

    return render_template("index.html", result=result, filename=filename)

@app.route('/submit-code', methods=['POST'])
def submit_code():
    code = request.form.get('code')
    result = None

    if code:
        code = code.strip().lower().replace(" ", "_")
        if code in recyclable:
            result = f"Code: {code}\n✅ Recyclable"
        elif code in conditionally_recyclable:
            result = f"Code: {code}\n✅ Conditionally Recyclable (requires special facilities)"
        elif code in compostable:
            result = f"Code: {code}\n🍂 Compostable"
        else:
            result = f"Code: {code}\n❌ Not recognized or not recyclable"

        with open("classification_log.txt", "a") as log_file:
            log_file.write(f"{datetime.now()} - Code Entry - {result}\n")

    return render_template("index.html", result=result)

@app.route('/logs')
def view_logs():
    try:
        with open("classification_log.txt", "r") as f:
            log_data = f.read()
    except FileNotFoundError:
        log_data = "No logs found."

    return f"<h1>Classification Logs</h1><pre>{log_data}</pre>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)