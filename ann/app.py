from flask import Flask, render_template, request, jsonify, abort
from flask_socketio import SocketIO, emit
import json
from reportprocessor import Processor
import os
import threading
from timer import TimerStorage
import torch

processor = Processor()

app = Flask(__name__)
socketio = SocketIO(app)

output_id_counter = 0  # Access the global counter

source_dir = 'demo' # TODO CHANGE TO YOUR DIRECTORY, also change in run_preprocess.py
PROJECT_ROOT = os.path.abspath(".") 
pipeline = None
training_thread = None
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['DATA_PATH'] = f'./preprocessing/{source_dir}/train_edited'
NOTES_DIR = os.path.join(app.config['DATA_PATH'], 'saved_notes')
app.secret_key = "your_secrete_key"
seed = 1
def safe_path(path):
    full_path = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if full_path.startswith(PROJECT_ROOT):
        return full_path
    else:
        abort(403)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/list_directory')
def list_directory():
    # Get the directory to query, default is empty meaning root directory
    dir_path = request.args.get('dir', '')
    abs_path = safe_path(dir_path)
    items = []
    for name in sorted(os.listdir(abs_path)):
        item_path = os.path.join(abs_path, name)
        if name.startswith('.'): # do not show hidden files or directories
            continue
        if os.path.isdir(item_path):
            items.append({"name": name, "type": "directory", "path": os.path.join(dir_path, name)})
        else:
            # If you only want to select .txt files, filter them here
            if name.lower().endswith('.txt'):
                filetype = "file"
                if os.path.exists(os.path.join(abs_path, name.rsplit('.', 1)[0] + '.json')):
                    filetype = "file_annotated"
                items.append({"name": name, "type": filetype, "path": os.path.join(dir_path, name)})
    
    return jsonify(items)

@app.route('/get_file_by_index', methods=['GET'])
def get_file_by_index():
    dir_param = request.args.get('dir', '')
    try:
        index = int(request.args.get('index', 0))
    except ValueError:
        return jsonify({'error': 'Invalid index'}), 400

    abs_dir = safe_path(dir_param)
    try:
        # Get all .txt files in the directory
        files = [f for f in os.listdir(abs_dir) if f.lower().endswith('.txt')]
        files.sort()
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    if index < 0 or index >= len(files):
        return jsonify({'error': 'Index out of range'}), 400

    filename = files[index]
    file_path = os.path.join(abs_dir, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Attempt to read the JSON annotation file with the same name
    json_filename = os.path.splitext(filename)[0] + '.json'
    json_path = os.path.join(abs_dir, json_filename)
    annotations = []
    relationships = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                annotations = data.get('annotations', [])
                relationships = data.get('relationships', [])
        except Exception as e:
            print("Error reading JSON:", e)

    return jsonify({
        'filename': filename,
        'content': content,
        'annotations': annotations,
        'relationships': relationships
    })

@app.route('/get_file/<path:filename>')
def get_file(filename):
    print("File name: \n", filename)
    abs_file = safe_path(filename)
    print("abs file: ", abs_file)
    if os.path.exists(abs_file):
        with open(abs_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    else:
        abort(404)


@app.route('/annotations', methods=['POST'])
def save_annotations():
    data = request.json

    # Get complete annotation data and relationship data from frontend
    annotations_data = data.get('annotations', [])
    relationships_data = data.get('relationships', [])
    txt_filename = data.get('filename', 'default.txt')

    # Final output JSON format: includes annotations and relationships
    json_output = {
         "annotations": annotations_data,
         "relationships": relationships_data
    }

    # Generate JSON filename
    json_filename = os.path.splitext(txt_filename)[0] + '.json'

    with open(json_filename, 'w', encoding='utf-8') as f:
         json.dump(json_output, f, ensure_ascii=False, indent=2)

    return jsonify({'message': f'Annotations saved successfully as {os.path.basename(json_filename)}!'})

@app.route('/general_annotations', methods=['POST'])
def save_general_annotations():
    data = request.json

    # Get complete annotation data and relationship data from frontend
    annotations_data = data.get('annotations', [])
    relationships_data = data.get('relationships', [])
    json_filename = data.get('filename', 'default.txt')

    # Final output JSON format: includes annotations and relationships
    json_output = {
         "annotations": annotations_data,
         "relationships": relationships_data
    }

    # Generate JSON filename
    json_file_path = os.path.splitext(json_filename)[0] + '.json'
    #json_file_path = os.path.join(app.config['DATA_PATH'], json_filename)

    with open(json_file_path, 'w', encoding='utf-8') as f:
         json.dump(json_output, f, ensure_ascii=False, indent=2)
    print("json_file_path: ", json_file_path)
    #shutil.copy(json_file_path, os.path.join(app.config['UPLOAD_FOLDER'], json_filename))
    return jsonify({'message': f'Annotations saved successfully as {json_filename}!'})

@socketio.on('start_training')
def handle_start_training(data):
    """
    Start model training and actively push data that needs annotation to the frontend.
    """
    global training_thread
    global pipeline
    # If training is already in progress, reject starting a new training
    if training_thread and training_thread.is_alive():
        socketio.emit('training_status', {'status': 'error', 'message': 'Training is already running.'})
        return

    force_start = data.get("force", False)
    # warning if previous training state found
    results_dir = 'datasets/results/scratch/CNBSE'
    if os.path.exists(results_dir) and not force_start:
        emit('training_status', {'status': 'warning', 'message': 'Previous training state found. Are you sure you want to start over? Clicking "OK" will delete all annotated data.'})
        return

    def run_training():
        pipeline.end_training = False
        pipeline.activelearning(strategyname='CNBSE', start_echo=0, stop_echo=150, seed=seed, resume=False,
                                query_prop=0.01, initial_prop=0.03, choices_number=5, NBest = 3, change_loss_threshold = 0.005, pretrained_model_path="models/model_pseudo_512.pth")
        # copy important csv files to directory
        import shutil
        shutil.copy(results_dir+'/seed_1_Training_losses.csv', app.config['DATA_PATH']+'/../')
        shutil.copy(results_dir+'/seed_1_Annotations.csv', app.config['DATA_PATH']+'/../')
        socketio.emit('training_status', {'status': 'completed', 'end_training': pipeline.end_training , 'message': 'Training completed.'})

    # reset timer and erase all annotations
    timer_dir = app.config['DATA_PATH']
    timer_file = os.path.join(timer_dir, 'timer_al.json')
    try:
        os.remove(timer_file)
    except:
        pass
    import shutil, glob
    shutil.rmtree(results_dir, ignore_errors=True)
    # remove all annotated jsons
    json_files = glob.glob(os.path.join(app.config['DATA_PATH'], "*.json"))
    for file in json_files:
        os.remove(file)
    threading.Thread(target=run_training).start()
    emit('training_status', {'status': 'running', 'mode': 'start', 'message': 'Training started.'})

@socketio.on('continue_training')
def handle_continue_training():
    """
    繼續先前中斷的訓練，從最後保存的狀態繼續。
    """
    global training_thread
    global pipeline

    # If training is already in progress, reject starting a new training
    if training_thread and training_thread.is_alive():
        socketio.emit('training_status', {'status': 'error', 'message': 'Training is already running.'})
        return

    # check if checkpoint exist
    if not os.path.exists(f'datasets/results/scratch/CNBSE/seed_{seed}_checkpoint.pkl'):
        emit('training_status', {'status': 'error', 'message': 'Previous training state not found. Please do `Start Training` again.'})
        return

    # Then start training in a new thread
    def run_training():
        pipeline.end_training = False
        torch.cuda.empty_cache()
        pipeline.activelearning(strategyname='CNBSE', start_echo=0, stop_echo=150, seed=seed, resume=True,
                                query_prop=0.01, initial_prop=0.3, choices_number=5, NBest = 3, change_loss_threshold = 0.005, pretrained_model_path="models/model_pseudo_512.pth")
        import shutil
        results_dir = 'datasets/results/scratch/CNBSE'
        shutil.copy(results_dir+'/seed_1_Training_losses.csv', app.config['DATA_PATH']+'/../')
        shutil.copy(results_dir+'/seed_1_Annotations.csv', app.config['DATA_PATH']+'/../')
        socketio.emit('training_status', {'status': 'completed', 'end_training': pipeline.end_training , 'message': 'Training completed.'})

    threading.Thread(target=run_training).start()
    emit('training_status', {'status': 'running', 'mode': 'continue', 'message': 'Training continued from previous state.'})

@app.route('/save_file_time', methods=['POST'])
def save_file_time():
    data = request.json
    filename = data.get('filename')
    annotation_time = data.get('annotation_time', {})
    # Read existing timer data
    timer_dir = app.config['DATA_PATH']
    os.makedirs(timer_dir, exist_ok=True)
    timer_file = os.path.join(timer_dir, 'timer_al.json')

    # Attempt to read existing timer data
    timer_data = {
        "file_times": []
    }

    try:
        if os.path.exists(timer_file):
            with open(timer_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if "file_times" in existing_data:
                    timer_data["file_times"] = existing_data["file_times"]
    except Exception as e:
        print(f"Error reading existing timer data: {e}")
    
    # Check if the file already exists in the list
    
    # If the file does not exist, add a new record
    if filename:
        timer_data["file_times"].append({
            "filename": filename,
            "time": annotation_time
        }) # add it to the back even if it may have existed already, will accumulate all time spent on the file
    
    # Save the updated data
    try:
        with open(timer_file, 'w', encoding='utf-8') as f:
            json.dump(timer_data, f, ensure_ascii=False, indent=2)
        print(f"File annotation time saved to {timer_file}")
    except Exception as e:
        print(f"Error saving file annotation time: {e}")
        return jsonify({'error': str(e)}), 500
    return jsonify({'message': 'File annotation time saved successfully!'})

@app.route('/get_resume_time', methods=['GET'])
def get_resume_time():
    timer_dir = app.config['DATA_PATH']
    os.makedirs(timer_dir, exist_ok=True)
    timer_file = os.path.join(timer_dir, 'timer_al.json')
    with open(timer_file) as f:
        data = json.load(f)
    file_times = data['file_times']
    latest = {'elapsedTime': file_times[-1]['time']}
    return jsonify(latest)


# For general annotation
@app.route('/save_timer', methods=['POST'])
def save_timer():
    data = request.json
    file_name = data.get('file').split('.txt')[0]
    elapsed_time = data.get('elapsed_time')
    response_data = TimerStorage.save_timer_state(file_name, elapsed_time, './preprocessing/')
    return jsonify({'message': 'Timer record saved successfully!'})

# For general annotation
@app.route('/get_timer', methods=['GET'])
def get_timer():
    file_name = request.args.get('file').split('.txt')[0]
    response_data = TimerStorage.get_timer_state(file_name, './preprocessing/')
    print(f"Returning timer data: {response_data}")
    return jsonify(response_data)


@app.route('/save_notes', methods=['POST'])
def save_notes():
    data = request.json
    filename = data.get('filename')
    notes = data.get('notes')
    current_dir = data.get('current_dir', None)
    if current_dir:
        notes_dir = os.path.join(current_dir, 'saved_notes')
    else:
        notes_dir = NOTES_DIR

    if not filename or not notes:
        return jsonify({'message': 'Invalid data'}), 400
    
    os.makedirs(notes_dir, exist_ok=True)
    file_path = os.path.join(notes_dir, filename)
    with open(file_path, 'w') as f:
        f.write(notes)
    return jsonify({'message': 'Notes saved successfully!'})

@app.route('/get_notes', methods=['GET'])
def get_notes():
    filename = request.args.get("filename")
    current_dir = request.args.get('current_dir', None)
    if current_dir:
        notes_dir = os.path.join(current_dir, 'saved_notes')
    else:
        notes_dir = NOTES_DIR

    if not filename:
        return jsonify({'message': 'Filename not provided'}), 400
    file_path = os.path.join(notes_dir, filename)
    if not os.path.exists(file_path):
        return jsonify({'message': 'No saved notes found', 'notes': ''})
    with open(file_path) as f:
        notes = f.read()
    return jsonify({'message': 'Notes retrieved successfully!', 'notes': notes})


if __name__ == "__main__":
    from AL_pipeline import BertALPipeline
    work_dir = os.path.abspath('./datasets')
    trainfile = os.path.join(work_dir,'Tags_training_512_cluster10.csv')
    testfile = None
    modelcard = 'emilyalsentzer/Bio_ClinicalBERT'
    pipeline = BertALPipeline(trainfile=trainfile, testfile=testfile, workdir=work_dir, modelcard=modelcard, socketio=socketio, data_folder=source_dir)
    socketio.run(app, port=8000, debug=True)

