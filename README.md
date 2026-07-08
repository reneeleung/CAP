## Clinical Annotation Platform (CAP)

A Human-in-the-Loop (HITL) framework for clinical text annotation utilizing dynamic active learning.

### Preprocess
First, place all notes (`.txt`) under `preprocessing/[data_dir_name]/train_edited/`. For demonstration purposes, 30 sample notes are under `preprocessing/sample/train_edited/` (these notes were extracted from the SLE dataset on [Zenodo](https://zenodo.org/records/20042444)).

**Token limit:** CAP uses BioClinicalBERT, which has a maximum input length of 512 tokens. All demo files are under this limit.

Then run the following, changing `source_dir` to your `[data_dir_name]`:
```bash
cd preprocessing/
python run_preprocess.py
```

### Run CAP
After preprocessing the notes, serve the application:
```bash
python app.py
```

#### Customizing the cold start
By default, CAP selects a small number of initial documents for annotation before making pre-annotations.

- For faster initial predictions, download a finetuned model and place it in `models`. For this demo, download from (here)[https://huggingface.co/reneeleung/sledai-annotation/blob/main/model_pseudo_512.pth].

- If you prefer to start without a finetuned model, increase the number of cold start samples by adjusting `initial_prop` in `app.py` (e.g. set to `0.3` for ~30% of the dataset).

### Exporting annotations
Annotations are saved in JSON format during the annotation process. After completing annotations, convert them to BRAT-style `.ann` files using the provided script:
```bash
cd demo/
python json_convert_ann.py --json_dir train_edited/ --txt_dir train_edited/ --output_dir train_edited/
```


### Multi-annotator setup
CAP is currently designed for single-annotator use per instance. To accommodate multiple annotators, we recommend cloning separate repositories for each annotator (e.g., cap-annotator-1/, cap-annotator-2/). Each clone maintains its own isolated database, preventing annotation conflicts. Run each on a different port (e.g., `:8000`, `:8001`) to serve simultaneously.
