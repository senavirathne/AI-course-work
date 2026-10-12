"""Build four small executable notebooks from the completed real study."""

from pathlib import Path
import nbformat as nbf

SETUP = """from pathlib import Path
import sys, json
PROJECT = Path.cwd()
if PROJECT.name == 'notebooks': PROJECT = PROJECT.parent
sys.path.insert(0, str(PROJECT / 'src'))
from fabric_inspector.common import read_json
from IPython.display import display, Image as DisplayImage, Markdown
import pandas as pd
def read(path): return read_json(PROJECT / path)
def figure(path): display(DisplayImage(filename=str(PROJECT / path)))
"""


def save(name, cells):
    notebook = nbf.v4.new_notebook(
        cells=[nbf.v4.new_markdown_cell(cells[0]), nbf.v4.new_code_cell(SETUP)]
    )
    notebook.cells.extend(
        nbf.v4.new_markdown_cell(s) if kind == "md" else nbf.v4.new_code_cell(s)
        for kind, s in cells[1:]
    )
    notebook.metadata.kernelspec = {
        "display_name": "Fabric coursework",
        "language": "python",
        "name": "fabric-coursework",
    }
    nbf.validate(notebook)
    nbf.write(notebook, Path("notebooks") / name)


def main():
    save(
        "01_dataset_and_eda.ipynb",
        [
            "# Q1 - Verified RAW-FABRID data and train-only EDA\nThe lecturer notebooks inspire loading, visualisation and tensor inspection. All assessed results use RAW-FABRID, never MNIST/CIFAR. This notebook reads the immutable full study and verifies the included authentic sample without requiring the 4 GB download.",
            (
                "code",
                "from fabric_inspector.data.manifest import REAL\nfull = read('data/manifests/split.json')\nassert full['provenance'] == REAL\nframe = pd.read_csv(PROJECT/'data/manifests/split.csv')\nassert len(frame)==19852\ndisplay(frame.groupby('split').agg(patches=('patch_id','size'), defective=('label','sum'), sources=('source_id','nunique')))\nprint(full['split_sha256'])",
            ),
            (
                "code",
                "for key in ['source_id','group_id','verified_extra_group']:\n    assert frame.groupby(key)['split'].nunique().max()==1\nprint('Source, duplicate-component and publisher-origin separation passed')\ndisplay(read('data/manifests/data-audit.json'))",
            ),
            (
                "code",
                "from fabric_inspector.data.manifest import load_manifest\nsample, rows, root = load_manifest(PROJECT/'data/sample-manifests/split.json', verify_files=True)\nassert len(rows)==168\nprint(sample['provenance'], len(rows), 'authentic engineering images; no sample performance claim')",
            ),
            (
                "code",
                "display(read('docs/eda-statistics.json'))\nfigure('reports/figures/eda-summary.png')\nfigure('reports/figures/eda-training-examples.png')\ndisplay(read('docs/eda-modelling-findings.json'))",
            ),
            (
                "md",
                "Three constraints: low defect prevalence requires recall-sensitive F2 and weighting; tiny regions require unchanged 256x256 detail; four training origin groups require modest capacity and group isolation. Exact-white mask foreground matches all 687 CSV areas. Origin IDs do not establish physical roll/manufacturer identities.",
            ),
        ],
    )
    save(
        "02_ann_and_optimisation.ipynb",
        [
            "# Q2-Q3 - Actual architectures and empirical optimisation\nInspect the measured study without rerunning selection or touching the held-out set.",
            (
                "code",
                "bundle=read('artifacts/serving-bundle.json')\ninitial=read('experiments/ann-search-000.json')\nfinal=read('experiments/'+bundle['run_id']+'.json')\nfor run in [initial,final]:\n    print(run['run_id'], run['architecture']['total_parameters'], 'parameters')\n    display(pd.DataFrame(run['architecture']['layers']))",
            ),
            (
                "code",
                "log=pd.read_csv(PROJECT/'reports/experiment-log.csv')\ndisplay(log)\nfigure('reports/figures/initial-final-learning.png')\nfigure('reports/figures/hyperparameter-study.png')",
            ),
            (
                "code",
                "failures=read('docs/failed-approaches.json')['attempts']\nassert len(failures)>=4 and sum(bool(f['substantive']) for f in failures)>=2\nfor case in failures:\n    display(Markdown('**'+case['run_id']+'**'))\n    display(case)\ndisplay(read('artifacts/seed-variability.json'))",
            ),
            (
                "code",
                "comparison=read('reports/final-evaluation/comparison.json')\nassert comparison['status']=='COMPLETE'\ndisplay(pd.DataFrame([{'model':r['model'],**r['metrics']} for r in comparison['comparisons']]))\ndisplay(read('reports/error-analysis.json'))",
            ),
            (
                "md",
                "The search covers learning rate, batch size, actual convolutional depth and AdamW versus momentum SGD. Numerical effects are bounded by the four-epoch screening budget. The complete logs include all unsuccessful measured runs and separate engineering interruptions. The family comparison is reported without changing the frozen split or choosing another seed on test scores.",
            ),
        ],
    )
    save(
        "03_serving_and_explainability.ipynb",
        [
            "# Q4-Q5 - Fresh inference and genuine errors\nThe same selected checkpoint and preprocessing serve the API and Grad-CAM.",
            (
                "code",
                "import torch, os\nos.environ['FABRIC_CPU_THREADS']='1'\ntorch.set_num_threads(1)\nfrom fabric_inspector.serving.predictor import Predictor\nfrom fabric_inspector.data.manifest import load_manifest\npredictor=Predictor(PROJECT/'artifacts/serving-bundle.json')\nsample,rows,root=load_manifest(PROJECT/'data/sample-manifests/split.json',verify_files=True)\nrow=next(r for r in rows if r['split']=='val')\nresult=predictor.predict((root/row['image_path']).read_bytes())\ndisplay(result)",
            ),
            (
                "code",
                "from PIL import Image\nfrom fabric_inspector.data.images import preprocess\nfrom fabric_inspector.explainability.gradcam import gradcam,overlay,occlusion_check\nimage=Image.open(root/row['image_path']).convert('L')\ntensor=preprocess(image,predictor.metadata['preprocessing']).unsqueeze(0).to(predictor.device)\ncam=gradcam(predictor.model,tensor)\nassert cam.shape==(256,256)\ndisplay(overlay(image,cam))\ndisplay(occlusion_check(predictor.model,tensor,cam))",
            ),
            (
                "code",
                "cases=read('docs/three-xai-errors.json')['cases']\nassert len(cases)>=3 and all(c['incorrect'] and c['label']!=c['prediction'] for c in cases)\nfor case in cases:\n    display(Markdown('**'+case['case_name']+'** - '+case['analysis']+'\\n\\nVisual inspection: '+case['visual_analysis']))\n    figure(case['panel'])\n    display(case['occlusion'])",
            ),
            (
                "code",
                "verification=read('reports/application-verification/results.json')\nassert verification['status']=='PASS'\ndisplay(verification)",
            ),
            (
                "md",
                "Grad-CAM is a coarse defect-logit sensitivity map; it is not a segmentation output or causal proof. Equal-area top/low/random mean-fill occlusion tests sensitivity. Negative drops, zero CAMs and failures against controls are reported. Integrated Gradients depends on a reference image; LIME depends on perturbations and a local surrogate.",
            ),
        ],
    )
    save(
        "04_real_vlm_and_final_protocol.ipynb",
        [
            "# Q6 - Real VLM responses and the frozen test protocol\nThese are genuine local SmolVLM generations, not simulated model outputs. This notebook only inspects completed evidence.",
            (
                "code",
                "from fabric_inspector.genai.vlm import parse_decision\nselection=read('artifacts/vlm-selection.json')\ndisplay(selection)\nfor index in [0,1]:\n    run=read(f'experiments/vlm-validation-{index}.json')\n    assert len(run['examples'])==3472\n    display({'prompt':run['config']['prompt'],'metrics':run['metrics'],'failure_rate':run['failure_rate']})\n    display(run['examples'][:2])",
            ),
            (
                "code",
                "frozen=read('artifacts/final-evaluation-manifest.json')\nguard=read('reports/final-evaluation/test-access.json')\nvlm=read('reports/final-evaluation/vlm.json')\nassert guard['status']=='COMPLETE'\nassert [e['patch_id'] for e in vlm['examples']]==frozen['test_patch_ids']\nassert len(vlm['examples'])==2996\nassert vlm['labels_used_only_for_scoring']\ndisplay(vlm['metrics'])\nprint('All models share 2996 frozen test patches; unresolved generations remain in denominators.')",
            ),
            (
                "code",
                "display(read('reports/final-evaluation/comparison.json')['test_seed_repetitions'])\ndisplay(read('reports/application-verification/results.json')['status'])\ndisplay(Markdown((PROJECT/'docs/ai-use-declaration.md').read_text()))",
            ),
            (
                "md",
                "No foundation-model class probability or class-only AUC is invented. Record the immutable revision, exact prompts, processor settings, token counts, raw outputs and local compute costs. A runtime reset lost the first completed final files; the identical frozen protocol was physically recomputed to reconstruct them, with no retraining or changed model/threshold/prompt/split. See docs/engineering-runs/final-results-workspace-reset.json. This is one benchmark, not independent repeated validation. Review the final report and viva guide: individual understanding and the live presentation must be demonstrated by the student.",
            ),
        ],
    )
    print("Built four evidence-based notebooks")


if __name__ == "__main__":
    main()
