import json
import re
import subprocess
import logging
import sys
import requests
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("/home/ubuntu/pipeline_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_filename = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)
logger.handlers.clear()

_file_handler = logging.FileHandler(log_filename)
_stream_handler = logging.StreamHandler(sys.stdout)
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_file_handler.setFormatter(_formatter)
_stream_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)


BASE_DIR      = Path("/home/ubuntu")
MATTERGEN_DIR = BASE_DIR / "mattergen"
OUTPUT_DIR    = BASE_DIR / "generated_structures"
PROMPTS_DIR   = BASE_DIR / "saved_prompts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)



GROQ_API_KEY = "PASTE_YOUR_GROQ_KEY_HERE"

# STEP 1 — GROQ API (Free LLaMA 3.3 70B)
def generate_hypothesis(user_description: str) -> dict:
    """
    Sends user description to Groq API using LLaMA 3.3 70B.
    Returns a structured dict with hypothesis fields.
    """
    logger.info("STAGE 1: Sending description to Groq API (LLaMA 3.3 70B)...")

    # Save the raw user prompt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_file = PROMPTS_DIR / f"user_prompt_{timestamp}.txt"
    prompt_file.write_text(user_description)
    logger.info(f"User prompt saved to: {prompt_file}")

    system_prompt = """You are a materials science expert working with generative diffusion models.
When given a material description, you MUST respond ONLY in this exact JSON format with no extra text:

{
  "chemical_composition": "e.g. Fe2O3 or TiO2 with dopants",
  "chemical_system": "Hyphen-separated list of 2-4 element symbols to condition the generative model on, e.g. 'Zr-B-Si-O' or 'Li-Fe-P-O'. Pick the elements most central to achieving the requested properties.",
  "crystal_structure": "e.g. cubic, hexagonal, orthorhombic",
  "key_properties": ["property1", "property2", "property3"],
  "constraints": ["constraint1", "constraint2"],
  "diffusion_prompt": "A single concise sentence for the diffusion model describing the material"
}

The "chemical_system" field is critical: it will be passed directly to a chemical-system-conditioned diffusion model, so it MUST be a valid hyphen-separated list of element symbols only (no compounds, no percentages, no extra text), e.g. "Zr-B-Si-O" not "ZrB2 with SiC".

Do NOT include markdown, code fences, or any explanation outside the JSON."""

    user_prompt = f"Design a new material based on this description:\n\n{user_description}"

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    try:
        logger.info("Calling Groq API...")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"].strip()
        logger.info("Received response from Groq.")

    except requests.exceptions.HTTPError as e:
        logger.error(f"Groq API HTTP error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Groq API error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Groq API connection error: {e}")
        raise RuntimeError(f"Could not connect to Groq API: {e}")

    # Clean response — strip markdown fences if present
    if "```" in raw_text:
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    # Parse structured JSON
    try:
        hypothesis = json.loads(raw_text)
        logger.info("Groq output successfully parsed as structured JSON.")
    except json.JSONDecodeError:
        logger.warning("Groq response was not valid JSON. Wrapping as plain hypothesis.")
        hypothesis = {
            "chemical_composition": "unknown",
            "crystal_structure": "unknown",
            "key_properties": [],
            "constraints": [],
            "diffusion_prompt": raw_text
        }

    # Save hypothesis to file
    hypothesis_file = PROMPTS_DIR / f"hypothesis_{timestamp}.json"
    hypothesis_file.write_text(json.dumps(hypothesis, indent=2))
    logger.info(f"Hypothesis saved to: {hypothesis_file}")
    logger.info(f"Hypothesis:\n{json.dumps(hypothesis, indent=2)}")

    return hypothesis



# CHEMICAL SYSTEM SANITIZATION
_VALID_ELEMENTS = {
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu"
}

def _sanitize_chemical_system(chem_sys: str) -> str:
    """
    Validates/cleans a chemical_system string to 'El-El-El' format
    using valid element symbols. Falls back to a safe default if invalid.
    """
    if not chem_sys:
        return "Li-Fe-O"

    candidates = re.split(r"[-,\s]+", chem_sys.strip())
    elements = [c for c in candidates if c in _VALID_ELEMENTS]

    seen = set()
    cleaned = []
    for el in elements:
        if el not in seen:
            seen.add(el)
            cleaned.append(el)

    if len(cleaned) < 2:
        return "Li-Fe-O"

    if len(cleaned) > 4:
        cleaned = cleaned[:4]

    return "-".join(cleaned)



# STEP 2 — MATTERGEN DIFFUSION MODEL
def run_mattergen(hypothesis: dict) -> Path:
    """
    Passes the diffusion_prompt from the hypothesis into MatterGen.
    Returns the absolute path to the output directory.
    """
    logger.info("STAGE 2: Running MatterGen diffusion model...")

    diffusion_prompt = hypothesis.get("diffusion_prompt", "")
    if not diffusion_prompt:
        logger.error("No diffusion_prompt found in hypothesis.")
        raise ValueError("Hypothesis is missing 'diffusion_prompt' field.")

    logger.info(f"Diffusion prompt: {diffusion_prompt}")

    chem_sys_raw = hypothesis.get("chemical_system", "")
    chemical_system = _sanitize_chemical_system(chem_sys_raw)
    if chem_sys_raw and chemical_system == "Li-Fe-O" and chem_sys_raw != "Li-Fe-O":
        logger.warning(
            f"Could not parse a valid chemical_system from hypothesis "
            f"('{chem_sys_raw}'). Falling back to default: {chemical_system}"
        )
    logger.info(f"Chemical system used for conditioning: {chemical_system}")

    run_output_dir = OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"MatterGen output will be saved to: {run_output_dir}")

    properties_arg = "{'chemical_system': '" + chemical_system + "'}"

    try:
        result = subprocess.run(
            [
                "mattergen-generate",
                str(run_output_dir),
                "--pretrained-name=chemical_system",
                "--batch_size=1",
                "--num_batches=1",
                f"--properties_to_condition_on={properties_arg}",
                "--diffusion_guidance_factor=2.0"
            ],
            cwd=str(MATTERGEN_DIR),
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            logger.error(f"MatterGen stderr:\n{result.stderr}")
            raise RuntimeError(f"MatterGen exited with code {result.returncode}")

        logger.info("MatterGen completed successfully.")
        logger.info(f"MatterGen stdout:\n{result.stdout}")

    except subprocess.TimeoutExpired:
        logger.error("MatterGen timed out after 10 minutes.")
        raise RuntimeError("MatterGen process timed out.")
    except FileNotFoundError:
        logger.error("mattergen-generate command not found.")
        raise RuntimeError("MatterGen not found. Make sure you ran: conda activate mattergen")

    return run_output_dir



# STEP 3 — CIF DETECTION, VALIDATION & RENDER

def render_structure(output_dir: Path, hypothesis: dict):
    """
    Finds, validates, and summarizes the generated .cif crystal structure file.
    Returns (cif_path, summary_dict)
    """
    logger.info("STAGE 3: Detecting and validating generated CIF file...")

    import zipfile
    zip_path = output_dir / "generated_crystals_cif.zip"

    if zip_path.exists():
        extract_dir = output_dir / "extracted_cif"
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        logger.info(f"Extracted CIF zip to: {extract_dir}")
        cif_files = list(extract_dir.rglob("*.cif"))
    else:
        cif_files = list(output_dir.rglob("*.cif"))

    if not cif_files:
        logger.error(f"No .cif files found under: {output_dir}")
        raise FileNotFoundError(f"No .cif file generated in {output_dir}")

    cif_path = cif_files[0]
    logger.info(f"CIF file found: {cif_path}")

    # CIF validation
    cif_size = cif_path.stat().st_size
    if cif_size == 0:
        raise ValueError(f"CIF file is empty: {cif_path}")

    cif_content = cif_path.read_text(errors="replace")
    if "_cell_length_a" not in cif_content:
        logger.warning("CIF file may be malformed — missing '_cell_length_a' marker.")
    else:
        logger.info("CIF file passed basic validation.")

    summary = {
        "formula": None,
        "num_atoms": None,
        "volume": None,
        "cif_path": str(cif_path),
        "crystal_structure": hypothesis.get("crystal_structure", "N/A"),
        "key_properties": hypothesis.get("key_properties", []),
    }

    # Structure summary using ASE
    try:
        from ase.io import read
        structure = read(str(cif_path))
        summary["formula"]   = structure.get_chemical_formula()
        summary["num_atoms"] = len(structure)
        summary["volume"]    = round(structure.get_volume(), 2)

        logger.info(f"Structure formula : {summary['formula']}")
        logger.info(f"Number of atoms   : {summary['num_atoms']}")
        logger.info(f"Cell volume       : {summary['volume']} Å³")

    except ImportError:
        logger.warning("ASE not installed. Run: pip install ase")
    except Exception as e:
        logger.error(f"Could not parse CIF with ASE: {e}")
        logger.info(f"CIF file is still saved at: {cif_path}")

    return cif_path, summary



# MAIN PIPELINE

def run_pipeline(user_description: str, progress_cb=None):
    """
    Runs the full pipeline for a given user description.

    progress_cb: optional callable(stage:int, message:str) called as the
                 pipeline progresses, useful for updating UI status.

    Returns a dict with: cif_path, summary, hypothesis, log_file
    """
    def notify(stage, msg):
        logger.info(msg)
        if progress_cb:
            try:
                progress_cb(stage, msg)
            except Exception:
                pass

    logger.info("=" * 60)
    logger.info("MATERIALS DESIGN PIPELINE — STARTED")
    logger.info("=" * 60)
    logger.info(f"Input description:\n{user_description.strip()}")

    try:
        notify(1, "Generating material hypothesis with LLaMA 3.3 70B (Groq)...")
        hypothesis = generate_hypothesis(user_description)

        notify(2, "Running MatterGen diffusion model on GPU (this may take a few minutes)...")
        output_dir = run_mattergen(hypothesis)

        notify(3, "Validating and parsing generated CIF structure...")
        cif_path, summary = render_structure(output_dir, hypothesis)

        notify(4, "Pipeline completed successfully.")
        logger.info("PIPELINE COMPLETED SUCCESSFULLY.")
        logger.info(f"Log saved to: {log_filename}")

        return {
            "cif_path": str(cif_path),
            "summary": summary,
            "hypothesis": hypothesis,
            "log_file": str(log_filename),
        }

    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}")
        logger.info(f"Check full log at: {log_filename}")
        raise



# CLI ENTRY POINT (still works standalone)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        description = " ".join(sys.argv[1:])
    else:
        description = input("Enter a description of the material you want to design:\n> ")
    run_pipeline(description)
