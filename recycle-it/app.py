from flask import Flask, request, render_template, redirect, url_for
import os
import subprocess
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Update these to match your model setup
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

            # Output file (just to avoid errors)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")

            # Run imagenet.py
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
                # Look for the classification output in stdout
                for line in output.splitlines():
                    if "class" in line.lower() and "%" in line:
                        result = line
                        if any(item in result for item in ["aluminum_soda_cans", "cardboard_packaging", "glass_beverage_bottles", "glass_cosmetic_containers", "glass_food_jars", "magazines", "newspaper", "office_paper", "paper_cups", "plastic_cup_lids", "plastic_detergent_bottles", "plastic_food_containers", "plastic_soda_bottles", "plastic_water_bottles", "steel_food_cans"]):
                            result = f"""{result}
                            ✅ Recyclable """

                        elif any(item in result for item in ["plastic_shopping_bags", "plastic_trash_bags", "plastic_straws", "styrofoam_cups", "styrofoam_food_containers", "disposable_plastic_cutlery"]):
                            result = f"""{result}
                            ✅ Conditionally Recyclable (requires special facilities)"""

                        elif any(item in result for item in ["coffee_grounds", "eggshells", "food_waste"]):
                            result = f"""{result}
                            🍂 Compostable """
                        break
                else:
                    result = "✅ Image processed, but no prediction found."
            except subprocess.CalledProcessError as e:
                result = f"❌ Error:\n{e.output}"

    return render_template("index.html", result=result, filename=filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
