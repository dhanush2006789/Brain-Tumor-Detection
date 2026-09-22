import os
import json
from flask import Flask, request, jsonify, send_from_directory, render_template, Response
from werkzeug.utils import secure_filename
import LoadUNet
import LoadSegNet
import testunet as tu
import testsegnet as ts
import accuracy as acc
import io
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
STATIC_FOLDER = 'static'

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load models once
unet_model = LoadUNet.load_model()
segnet_model = LoadSegNet.load_model()


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("No image provided")
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    unet_output_filename = f"unet_{filename}"
    segnet_output_filename = f"segnet_{filename}"
    unet_output_path = os.path.join(OUTPUT_FOLDER, unet_output_filename)
    segnet_output_path = os.path.join(OUTPUT_FOLDER, segnet_output_filename)

    print(f"Running prediction for: {image_path}")
    print(f"UNet output will be saved to: {unet_output_path}")
    print(f"SegNet output will be saved to: {segnet_output_path}")

    try:
        import shutil
        shutil.copy2(image_path, os.path.join(OUTPUT_FOLDER, filename))

        tu.predict_unet(unet_model, image_path, unet_output_path)
        ts.predict_segnet(segnet_model, image_path, segnet_output_path)

        #calculating accuracy
        unet_acc = acc.calculate_accuracy(unet_output_path, image_path)
        segnet_acc = acc.calculate_accuracy(segnet_output_path, image_path)

        # Load or initialize the accuracy data
        data_file = 'accuracy_data.json'
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                accuracy_data = json.load(f)
        else:
            accuracy_data = {"unet": [], "segnet": []}

        # Append new accuracy values
        accuracy_data['unet'].append(unet_acc)
        accuracy_data['segnet'].append(segnet_acc)

        # Save it back
        with open(data_file, 'w') as f:
            json.dump(accuracy_data, f)
        print(f"UNet accuracy: {unet_acc:.2f}")
        print(f"SegNet accuracy: {segnet_acc:.2f}")

        print("Prediction completed.")
        return jsonify({
            'original': f"/uploads/{filename}",
            'unet_output': f"/outputs/{unet_output_filename}",
            'segnet_output': f"/outputs/{segnet_output_filename}",
            'unet_acc': unet_acc,
            'segnet_acc': segnet_acc
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Log the real reason
        return jsonify({"error": str(e)}), 500

@app.route('/accuracy_plot.png')
def accuracy_plot():
    # Load accuracy data
    data_file = 'accuracy_data.json'
    if not os.path.exists(data_file):
        return "No accuracy data found", 404

    with open(data_file, 'r') as f:
        accuracy_data = json.load(f)

    unet_acc = accuracy_data.get('unet', [])
    segnet_acc = accuracy_data.get('segnet', [])

    # X-axis: number of inputs (1-based index)
    x = list(range(1, max(len(unet_acc), len(segnet_acc)) + 1))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x[:len(unet_acc)], unet_acc, label='UNet', color='blue', marker='o')
    ax.plot(x[:len(segnet_acc)], segnet_acc, label='SegNet', color='orange', marker='x')

    ax.set_xlabel('Number of Inputs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy over Inputs')

    # Place legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on right for legend

    # Save plot to PNG image in memory
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)  # close the figure to free memory
    buf.seek(0)

    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/accuracy_data')
def accuracy_data():
    import json
    with open('accuracy_data.json', 'r') as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)