import os, time, uuid
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from A2 import process_image
from matching_T import run_template_matching   # uses your working matching_T.py

app = Flask(__name__, static_folder='static', template_folder='templates')

#Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR  = os.path.join(BASE_DIR, 'static')
DATASET     = os.path.join(STATIC_DIR, 'dataset')
UPLOADS     = os.path.join(STATIC_DIR, 'uploads')
RESULTS     = os.path.join(STATIC_DIR, 'results')
TM_RESULTS  = os.path.join(STATIC_DIR, 'output_TM')  # template-matching outputs

os.makedirs(DATASET, exist_ok=True)
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(TM_RESULTS, exist_ok=True)

ALLOWED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def list_dataset():
    return [
        f for f in sorted(os.listdir(DATASET))
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTS
    ]


def is_safe_static_relpath(relpath: str) -> bool:
    if not relpath or '..' in relpath or relpath.startswith('/'):
        return False
    top = relpath.split('/')[0]
    return top in ('dataset', 'uploads')


def to_abs_static(relpath: str) -> str:
    assert is_safe_static_relpath(relpath)
    return os.path.join(STATIC_DIR, relpath.replace('/', os.sep))


def list_tm_results():
    if not os.path.exists(TM_RESULTS):
        return []
    return [
        f"output_TM/{f}"
        for f in sorted(os.listdir(TM_RESULTS))
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTS
    ]


# ROUTES

@app.route('/', methods=['GET'])
def index():
    # tab can be 'dataset', 'upload', or 'template'
    active_tab = request.args.get('tab', 'dataset')
    return render_template(
        'index.html',
        files=list_dataset(),
        image_path=None,
        results=None,
        params=None,
        active_tab=active_tab,
        error=None,
        tm_results=list_tm_results()
    )


@app.route('/view', methods=['GET'])
def view():
    rel = request.args.get('path', '')
    active_tab = request.args.get('tab', 'dataset')

    if not is_safe_static_relpath(rel):
        return render_template(
            'index.html',
            files=list_dataset(),
            image_path=None,
            results=None,
            params=None,
            active_tab='dataset',
            error='Invalid image path.',
            tm_results=list_tm_results()
        )

    return render_template(
        'index.html',
        files=list_dataset(),
        image_path=rel,
        results=None,
        params=None,
        active_tab=active_tab,
        error=None,
        tm_results=list_tm_results()
    )


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if not file or not file.filename:
        return render_template(
            'index.html',
            files=list_dataset(),
            image_path=None,
            results=None,
            params=None,
            active_tab='upload',
            error='No file selected.',
            tm_results=list_tm_results()
        )

    name = secure_filename(file.filename)
    _, ext = os.path.splitext(name)
    if ext.lower() not in ALLOWED_EXTS:
        return render_template(
            'index.html',
            files=list_dataset(),
            image_path=None,
            results=None,
            params=None,
            active_tab='upload',
            error=f'Unsupported file type: {ext}',
            tm_results=list_tm_results()
        )

    unique = f"{uuid.uuid4().hex}{ext.lower()}"
    save_path = os.path.join(UPLOADS, unique)
    file.save(save_path)

    return redirect(url_for('view', path=f'uploads/{unique}', tab='upload'))


@app.route('/restore', methods=['GET'])
def restore():
    rel = request.args.get('path', '')
    active_tab = request.args.get('tab', 'dataset')

    try:
        sigma = float(request.args.get('sigma', 7))
        K = float(request.args.get('K', 0.002))
        iterations = int(request.args.get('iterations', 8))
    except ValueError:
        return render_template(
            'index.html',
            files=list_dataset(),
            image_path=None,
            results=None,
            params=None,
            active_tab=active_tab,
            error='Bad parameters.',
            tm_results=list_tm_results()
        )

    if not is_safe_static_relpath(rel):
        return render_template(
            'index.html',
            files=list_dataset(),
            image_path=None,
            results=None,
            params=None,
            active_tab=active_tab,
            error='Pick an image first.',
            tm_results=list_tm_results()
        )

    abs_in = to_abs_static(rel)
    if not os.path.exists(abs_in):
        return render_template(
            'index.html',
            files=list_dataset(),
            image_path=None,
            results=None,
            params=None,
            active_tab=active_tab,
            error='Selected image does not exist.',
            tm_results=list_tm_results()
        )

    # Run Part 1 (Hybrid Restoration)
    process_image(abs_in, RESULTS, sigma=sigma)

    base = os.path.splitext(os.path.basename(abs_in))[0]
    out_name = f"{base}_comparison.png"
    out_rel  = f"results/{out_name}"

    results = [{
        "title": f"Hybrid Restoration (σ={sigma})",
        "file": out_rel,
        "v": int(time.time())
    }]
    params = {"sigma": sigma, "K": K, "iterations": iterations}

    return render_template(
        'index.html',
        files=list_dataset(),
        image_path=rel,
        results=results,
        params=params,
        active_tab=active_tab,
        error=None,
        tm_results=list_tm_results()
    )


@app.route('/template_match', methods=['POST'])
def template_match():
    """
    Part 2 – run matching_T on all test images and show results
    in the Template Matching tab.
    """
    tm_files = run_template_matching(show_plots=False)

    return render_template(
        'index.html',
        files=list_dataset(),
        image_path=None,
        results=None,
        params=None,
        active_tab='template',
        error=None,
        tm_results=tm_files
    )


if __name__ == '__main__':
    app.run(debug=True)
