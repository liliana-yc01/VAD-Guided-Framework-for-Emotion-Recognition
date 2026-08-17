# ==============================================================================
# PHASE 1: Emotional Data Mapping &
# PHASE 2: Mathematical Framework Formalization &
# PHASE 3: Algorithmic Logic Integration
# FULL EVALUATION PIPELINE: Baseline vs. Theory-Guided
# ==============================================================================
import os
import re
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------
# 1. Set up DeepSeek API Client
# ---------------------------------------------------------------------
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    # Optional fallback if environment variable is not set
    api_key = input("Enter your DeepSeek API Key: ").strip()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)
MODEL_NAME = "deepseek-v4-flash"

BATCH_SAVE_INTERVAL = 20  # Save progress to CSV every 20 API calls
CHECKPOINT_FILE = "emotion_llm_full_results.csv"

def save_checkpoint(df, path):
    """Save DataFrame to CSV atomically to avoid corruption on crash."""
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)

# ---------------------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
CSV_PATH = os.path.join(SCRIPT_DIR, "sample_data", "emotion-emotion_69k.csv")

df = pd.read_csv(CSV_PATH)

print("Dataset loaded.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# ---------------------------------------------------------------------
# 3. Clean data & Define Target Emotions
# ---------------------------------------------------------------------
# Clean emotion labels
df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()

# Clean situation text
df["Situation"] = df["Situation"].astype(str).str.strip()

# Standardized list of Ekman's basic emotions
target_emotions = ["angry", "afraid", "joyful", "sad"]
valid_emotions = target_emotions

# Continuous VAD Coordinates (Russell & Mehrabian, 1977 Table 4 normalized to [-1.0, +1.0])
vad_coords = {
    "angry": np.array([-0.51, 0.59, 0.25]),
    "afraid": np.array([-0.64, 0.60, -0.43]),
    "joyful": np.array([0.76, 0.48, 0.35]),
    "sad": np.array([-0.63, -0.27, -0.33]),
}

RM_WEIGHTS = (
    0.353, # w_V
    0.114, # w_A 
    0.228 # w_D
)

def closest_emotion(vad_vector, prototypes, weights=RM_WEIGHTS):
    """
    Calculates weighted Euclidean distance to each prototype:
    d = sqrt( w_V * (ΔV)^2 + w_A * (ΔA)^2 + w_D * (ΔD)^2 )
    
    Returns the key (emotion string) of the nearest prototype.
    """
    w_v, w_a, w_d = weights
    best_label = None
    min_dist = float("inf")

    for label, proto_vec in prototypes.items():
        diff = vad_vector - proto_vec
        # Weighted Euclidean distance
        dist = np.sqrt(
            w_v * (diff[0] ** 2) + 
            w_a * (diff[1] ** 2) + 
            w_d * (diff[2] ** 2)
        )
        if dist < min_dist:
            min_dist = dist
            best_label = label

    return best_label

# Filter dataset to target emotions
df_fine_clean = df[df["emotion"].isin(target_emotions)][
    ["Situation", "emotion", "empathetic_dialogues"]
].copy()

# Remove empty or very short situations
df_fine_clean = df_fine_clean[
    df_fine_clean["Situation"].fillna("").str.len() > 15
].copy()

print("\nAvailable examples per target emotion:")
print(df_fine_clean["emotion"].value_counts())

# ---------------------------------------------------------------------
# 4. Sample exactly 1940 instances per target emotion (1940 x 4 = 7760 samples)
# ---------------------------------------------------------------------
SAMPLES_PER_EMOTION = 1940
df_sample = (
    df_fine_clean.groupby("emotion", group_keys=False)
    .sample(n=SAMPLES_PER_EMOTION, random_state=42)
    .reset_index(drop=True)
)

print(f"\nSampled {SAMPLES_PER_EMOTION} instances per emotion.")
print(df_sample["emotion"].value_counts())
print("Total rows:", len(df_sample))

print("\nFirst few sample rows:")
print(df_sample[["Situation", "emotion"]].head())


# ---------------------------------------------------------------------
# 5. Prompt 1: Baseline Prompt
# ---------------------------------------------------------------------
def create_baseline_prompt(situation, valid_emotions_list):
    emotion_options = ", ".join(valid_emotions_list)

    prompt = f"""
Classify the emotion expressed in the following situation.

Choose exactly one emotion label from this list:
[{emotion_options}]

Situation:
\"{situation}\"

Return only the emotion label.
Emotion:
"""
    return prompt.strip()


# ---------------------------------------------------------------------
# 6. Prompt 2: Theory-Guided Prompt (Algorithmic VAD CoT)
# ---------------------------------------------------------------------
def create_theory_guided_prompt(situation, valid_emotions_list):
    emotion_options = ", ".join(valid_emotions_list)

    prompt = f"""
Analyze the emotion in the situation using the Valence-Arousal-Dominance (VAD) framework.

VAD Dimensions (-1.0 to +1.0):
- Valence (V): Negative/Unpleasant (-1.0) to Positive/Pleasant (+1.0)
- Arousal (A): Low/Calm/Passive (-1.0) to High/Excited/Active (+1.0)
- Dominance (D): Submissive/Vulnerable (-1.0) to Dominant/Assertive (+1.0)

Empirical VAD Prototypes:
- angry: [V: -0.51, A: +0.59, D: +0.25] -> High displeasure, high arousal, high/assertive control.
- afraid: [V: -0.64, A: +0.60, D: -0.43] -> High displeasure, high arousal, low/vulnerable control.
- joyful: [V: +0.76, A: +0.48, D: +0.35] -> High pleasure, high arousal, high control/empowerment.
- sad: [V: -0.63, A: -0.27, D: -0.33] -> High displeasure, low arousal, low control/dejection.

**CRITICAL RULE 1: PRIMACY OF EXPLICIT EMOTION WORDS**
If the speaker uses an explicit emotion word (e.g., “mad”, “scared”, “afraid”, “happy”, “sad”, “joyful”), that label MUST be your final prediction. Use VAD only to explain, not to override.

**CRITICAL RULE 2: TEMPORAL ANCHORING**
You MUST judge the emotion felt DURING the described event, NOT the speaker's current reflection on it.
- If the text says "I used to be terrified" → The emotion is TERRIFIED (Afraid), NOT Joyful.
- If the text says "I am nervous about the rollercoaster" → The emotion is NERVOUS (Afraid), NOT excited/Joyful.
- If the text says "I preordered a game but can't play until October" → The dominant feeling is DISAPPOINTMENT (Sad), NOT Joyful.

**CRITICAL RULE 3: ANGER vs. FEAR (The Blame Test)**
- ANGER: a specific person/entity is the cause, and the speaker feels frustrated/irritated and wants to confront or blame. Even if trapped, if the threat is not life‑threatening and the reaction is annoyance, it’s ANGER.
- FEAR: the threat is ambiguous, environmental, or life‑threatening, and the speaker wants to escape or hide. If there is no clear agent to blame, it’s FEAR.

**CRITICAL RULE 4: SADNESS vs. ANGER (The Energy Test)**
- SADNESS: usually low energy, passive, grief, longing, resignation, emptiness, or unchangeable melancholy.
- ANGER: high energy, irritation, frustration, complaining. Even if delivered in a resigned tone, active irritation = ANGER.

**CRITICAL RULE 5: THRILL & ANTICIPATION CONTEXTS**
- Mentioning inherently risky or intimidating activities (e.g., roller coasters, dangerous driving, public speaking) without positive words like "excited" or "love" reflects anxiety/dread -> afraid.

--- FEW-SHOT EXAMPLES ---

Situation: "My roommate ate the leftovers I was saving for dinner without asking."
REASONING: Negative event, high arousal (irritation), assertive control → anger.
VAD_Estimate: [V: -0.60, A: +0.65, D: +0.25]
EMOTION: angry

Situation: "I woke up in the middle of the night to the sound of glass shattering downstairs and footsteps approaching my bedroom door."
REASONING: Immediate, uncontrollable, unambiguous threat to personal safety, triggering panic and helplessness rather than confrontation → high displeasure, high arousal, low dominance, complete helplessness/vulnerability → fear.
VAD_Estimate: [V: -0.75, A: +0.65, D: -0.50]
EMOTION: afraid

Situation: "My mother just collapsed and the paramedics are trying to revive her."
REASONING: Sudden, traumatic, and devastating loss of a loved one triggering helpless shock and grief rather than self-preservation → strongly negative valence, high-arousal distress, vulnerable helplessness, low dominance → sad.
VAD_Estimate: [V: -0.80, A: +0.65, D: -0.50]
EMOTION: sad

Situation: "I just found out I got accepted into my dream university with a full scholarship."
REASONING: Unexpected, life-changing personal achievement triggering elation and pride → high pleasure, high excitement, strong sense of empowerment → joyful.
VAD_Estimate: [V: +0.85, A: +0.70, D: +0.45]
EMOTION: joyful

---

Task:
You are an expert in emotion analysis. Given a situation, you will:
1. Reason through the VAD dimensions based on the text.
2. Estimate its Valence (V), Arousal (A), and Dominance (D) on a scale from -1.0 to +1.0.
3. Compare your estimate to the benchmark coordinates above, output the closest benchmark emotion from this list: [{emotion_options}].
4. Make sure your final EMOTION label is consistent with your VAD estimate; if there is a conflict, re‑evaluate.

Format your output EXACTLY like this:
REASONING: <1-2 sentences of step-by-step VAD evaluation>
VAD_Estimate: [V: x.xx, A: x.xx, D: x.xx]
EMOTION: <exactly one label from the list>

Situation: "{situation}"
"""
    return prompt.strip()


# ---------------------------------------------------------------------
# 7. DeepSeek Call & Algorithmic VAD Vector Distance Parsing
# ---------------------------------------------------------------------
def parse_and_map_vad(raw_output, valid_emotions_list, vad_coords, weights=RM_WEIGHTS):
    if raw_output is None or not isinstance(raw_output, str):
        return "invalid_output"

    # STEP 1: ALWAYS check for the explicit EMOTION tag first (LLM's final decision)
    emotion_match = re.search(r"EMOTION:\s*([a-zA-Z]+)", raw_output, re.IGNORECASE)
    if emotion_match:
        predicted_tag = emotion_match.group(1).lower().strip()
        if predicted_tag in valid_emotions_list:
            return predicted_tag  # <-- Trust the LLM's explicit choice!

    # STEP 2: Fallback to VAD math ONLY if the EMOTION tag is missing/broken
    vad_match = re.search(
        r"VAD_Estimate:\s*\[V:\s*([+-]?\d*\.?\d+),\s*A:\s*([+-]?\d*\.?\d+),\s*D:\s*([+-]?\d*\.?\d+)\]",
        raw_output,
        re.IGNORECASE,
    )
    if vad_match:
        try:
            est = np.array([float(vad_match.group(1)), float(vad_match.group(2)), float(vad_match.group(3))])
            predicted = closest_emotion(est, vad_coords, weights=weights)
            if predicted in valid_emotions_list:
                return predicted
        except Exception:
            pass

    return "invalid_output"


def clean_prediction(raw_output, valid_emotions_list):
    """Cleans single-word outputs for baseline prompt."""
    if raw_output is None:
        return "invalid_output"

    output = raw_output.strip().lower()
    output = output.replace('"', "").replace("'", "").replace(".", "").replace(",", "").strip()

    if output in valid_emotions_list:
        return output

    found = [emo for emo in valid_emotions_list if re.search(rf"\b{emo}\b", output)]
    if len(found) == 1:
        return found[0]

    return "invalid_output"


def extract_vad_vector(raw_output):
    """Return estimated VAD vector as np.array, or None if not found."""
    if raw_output is None or not isinstance(raw_output, str):
        return None
    m = re.search(
        r"VAD_Estimate:\s*\[V:\s*([+-]?\d*\.?\d+),\s*A:\s*([+-]?\d*\.?\d+),\s*D:\s*([+-]?\d*\.?\d+)\]",
        raw_output, re.IGNORECASE
    )
    if m:
        try:
            return np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        except Exception:
            pass
    return None

def repair_invalid(raw_output, valid_emotions_list):
    """
    Ultimate fallback repair for invalid outputs.
    Scans raw text for explicit emotion words, VAD vector, or common synonyms.
    """
    if not isinstance(raw_output, str):
        return "invalid_output"

    # 1. Check for explicit EMOTION: tag (some LLM outputs may have it)
    emotion_match = re.search(r"EMOTION:\s*([a-zA-Z]+)", raw_output, re.IGNORECASE)
    if emotion_match:
        tag = emotion_match.group(1).lower().strip()
        if tag in valid_emotions_list:
            return tag

    # 2. Check for VAD vector and map it
    vad_match = re.search(
        r"\[V:\s*([+-]?\d*\.?\d+),\s*A:\s*([+-]?\d*\.?\d+),\s*D:\s*([+-]?\d*\.?\d+)\]",
        raw_output, re.IGNORECASE
    )
    if vad_match:
        try:
            est = np.array([float(vad_match.group(1)), float(vad_match.group(2)), float(vad_match.group(3))])
            predicted = closest_emotion(est, vad_coords, weights=RM_WEIGHTS)
            if predicted in valid_emotions_list:
                return predicted
        except:
            pass

    # 3. Keyword search for the 4 emotion words anywhere in the text
    for emo in valid_emotions_list:
        if re.search(rf'\b{emo}\b', raw_output.lower()):
            return emo

    # 4. Check for common synonyms
    synonym_map = {
        "furious": "angry", "irritated": "angry", "mad": "angry",
        "scared": "afraid", "terrified": "afraid", "nervous": "afraid",
        "happy": "joyful", "delighted": "joyful", "excited": "joyful",
        "depressed": "sad", "disappointed": "sad", "devastated": "sad"
    }
    for synonym, emo in synonym_map.items():
        if re.search(rf'\b{synonym}\b', raw_output.lower()):
            return emo

    return "invalid_output"

def call_deepseek(prompt, valid_emotions_list, is_theory=False, vad_coords_dict=None, max_retries=3, initial_backoff=4):
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.0
            )
            raw_output = response.choices[0].message.content
            
            if is_theory and vad_coords_dict is not None:
                prediction = parse_and_map_vad(raw_output, valid_emotions_list,
                                          vad_coords_dict)
            else:
                prediction = clean_prediction(raw_output, valid_emotions_list)
                
            if prediction == "invalid_output" and raw_output is not None:
                prediction = repair_invalid(raw_output, valid_emotions_list)

            time.sleep(0.5)
            return prediction, raw_output

        except Exception as e:
            print(f"\n[API Error - Attempt {attempt + 1}/{max_retries}]", e)
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                return "api_error", None

    return "api_error", None


# ---------------------------------------------------------------------
# 8. Run Baseline Evaluation
# ---------------------------------------------------------------------
baseline_predictions = []
baseline_raw_outputs = []

# Check if a checkpoint exists to resume progress
if os.path.exists(CHECKPOINT_FILE):
    try:
        df_ckpt = pd.read_csv(CHECKPOINT_FILE)
        if "baseline_prediction" in df_ckpt.columns and len(df_ckpt) == len(df_sample):
            valid_mask = df_ckpt["baseline_prediction"].notna()
            baseline_predictions = df_ckpt.loc[valid_mask, "baseline_prediction"].tolist()
            baseline_raw_outputs = df_ckpt.loc[valid_mask, "baseline_raw_output"].tolist()
            print(f"\n[Checkpoint Found] Resuming baseline evaluation from item {len(baseline_predictions) + 1}/{len(df_sample)}...")
    except Exception as e:
        print("\nCould not load existing baseline checkpoint, starting fresh:", e)

if len(baseline_predictions) < len(df_sample):
    print("\nStarting baseline prompt evaluation...\n")

for index in range(len(baseline_predictions), len(df_sample)):
    row = df_sample.iloc[index]
    situation = row["Situation"]
    prompt = create_baseline_prompt(situation, valid_emotions)

    prediction, raw_output = call_deepseek(prompt, valid_emotions, is_theory=False)

    baseline_predictions.append(prediction)
    baseline_raw_outputs.append(raw_output)

    print(f"Baseline item {index + 1}/{len(df_sample)}")
    print("True label:", row["emotion"])
    print("Prediction:", prediction)
    print("-" * 60)

    # Save batch checkpoint to disk
    if (index + 1) % BATCH_SAVE_INTERVAL == 0 or (index + 1) == len(df_sample):
        df_sample.loc[:index, "baseline_prediction"] = baseline_predictions
        df_sample.loc[:index, "baseline_raw_output"] = baseline_raw_outputs
        save_checkpoint(df_sample, CHECKPOINT_FILE)

df_sample["baseline_prediction"] = baseline_predictions
df_sample["baseline_raw_output"] = baseline_raw_outputs

# ---------------------------------------------------------------------
# 9. Run Theory-Guided Evaluation
# ---------------------------------------------------------------------
theory_predictions = []
theory_raw_outputs = []

# Check if a checkpoint exists for theory predictions
if os.path.exists(CHECKPOINT_FILE):
    try:
        df_ckpt = pd.read_csv(CHECKPOINT_FILE)
        if "theory_prediction" in df_ckpt.columns and len(df_ckpt) == len(df_sample):
            valid_mask = df_ckpt["theory_prediction"].notna()
            theory_predictions = df_ckpt.loc[valid_mask, "theory_prediction"].tolist()
            theory_raw_outputs = df_ckpt.loc[valid_mask, "theory_raw_output"].tolist()
            print(f"\n[Checkpoint Found] Resuming theory evaluation from item {len(theory_predictions) + 1}/{len(df_sample)}...")
    except Exception as e:
        print("\nCould not load existing theory checkpoint, starting fresh:", e)

if len(theory_predictions) < len(df_sample):
    print("\nStarting theory-guided prompt evaluation...\n")

for index in range(len(theory_predictions), len(df_sample)):
    row = df_sample.iloc[index]
    situation = row["Situation"]
    # Make sure to pass valid_emotions here
    prompt = create_theory_guided_prompt(situation, valid_emotions)

    # Use the call_deepseek wrapper to enforce proper parsing and retries
    prediction, raw_output = call_deepseek(prompt, valid_emotions, is_theory=True, vad_coords_dict=vad_coords)

    theory_predictions.append(prediction)
    theory_raw_outputs.append(raw_output)

    print(f"Theory-guided item {index+1}/{len(df_sample)}")
    print("True label:", row["emotion"])
    print("Prediction:", prediction)
    print("-" * 60)
    
    # Save batch checkpoint to disk
    if (index + 1) % BATCH_SAVE_INTERVAL == 0 or (index + 1) == len(df_sample):
        df_sample.loc[:index, "theory_prediction"] = theory_predictions
        df_sample.loc[:index, "theory_raw_output"] = theory_raw_outputs
        save_checkpoint(df_sample, CHECKPOINT_FILE)

# Assuming saved the estimated vectors in df_sample["theory_estimated_vec"]

df_sample["theory_prediction"] = theory_predictions
df_sample["theory_raw_output"] = theory_raw_outputs

# Extract VAD vectors from the raw outputs
df_sample["theory_estimated_vec"] = df_sample["theory_raw_output"].apply(extract_vad_vector)

# Define the function to compute VAD estimation error
def calculate_true_vad_error(row):
    """Distance between estimated VAD vector and true emotion’s benchmark coordinates."""
    true_label = row["emotion"]
    est_vec = row["theory_estimated_vec"]
    if true_label in vad_coords and isinstance(est_vec, np.ndarray):
        return float(np.linalg.norm(vad_coords[true_label] - est_vec))
    return np.nan

# Compute and print the VAD estimation error
df_sample["theory_true_vad_error"] = df_sample.apply(calculate_true_vad_error, axis=1)
print("Mean VAD estimation error to true benchmark:", df_sample["theory_true_vad_error"].mean())


# ---------------------------------------------------------------------
# 10. Evaluate Baseline Prompt
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("BASELINE PROMPT RESULTS (EKMAN 6)")
print("=" * 70)

baseline_accuracy = accuracy_score(
    df_sample["emotion"], df_sample["baseline_prediction"]
)

print(f"Baseline accuracy: {baseline_accuracy:.3f}")

print("\nBaseline classification report:")
print(
    classification_report(
        df_sample["emotion"],
        df_sample["baseline_prediction"],
        labels=target_emotions,
        zero_division=0,
    )
)

baseline_confusion = confusion_matrix(
    df_sample["emotion"], df_sample["baseline_prediction"], labels=target_emotions
)

baseline_confusion_df = pd.DataFrame(
    baseline_confusion,
    index=[f"true_{x}" for x in target_emotions],
    columns=[f"pred_{x}" for x in target_emotions],
)

print("\nBaseline confusion matrix:")
print(baseline_confusion_df)

# ---------------------------------------------------------------------
# 11. Evaluate Theory-Guided Prompt
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("THEORY-GUIDED PROMPT RESULTS")
print("=" * 70)

theory_accuracy = accuracy_score(
    df_sample["emotion"], df_sample["theory_prediction"]
)

print(f"Theory-guided accuracy: {theory_accuracy:.3f}")

print("\nTheory-guided classification report:")
print(
    classification_report(
        df_sample["emotion"],
        df_sample["theory_prediction"],
        labels=target_emotions,
        zero_division=0,
    )
)

theory_confusion = confusion_matrix(
    df_sample["emotion"], df_sample["theory_prediction"], labels=target_emotions
)

theory_confusion_df = pd.DataFrame(
    theory_confusion,
    index=[f"true_{x}" for x in target_emotions],
    columns=[f"pred_{x}" for x in target_emotions],
)

print("\nTheory-guided confusion matrix:")
print(theory_confusion_df)

# ---------------------------------------------------------------------
# 12. Compare Baseline vs. Theory-Guided Prompt
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print(f"Baseline accuracy:      {baseline_accuracy:.3f}")
print(f"Theory-guided accuracy: {theory_accuracy:.3f}")
print(f"Difference:             {theory_accuracy - baseline_accuracy:.3f}")

df_sample["baseline_correct"] = (
    df_sample["emotion"] == df_sample["baseline_prediction"]
)

df_sample["theory_correct"] = (
    df_sample["emotion"] == df_sample["theory_prediction"]
)

helped_cases = df_sample[
    (df_sample["baseline_correct"] == False)
    & (df_sample["theory_correct"] == True)
].copy()

hurt_cases = df_sample[
    (df_sample["baseline_correct"] == True)
    & (df_sample["theory_correct"] == False)
].copy()

both_wrong_cases = df_sample[
    (df_sample["baseline_correct"] == False)
    & (df_sample["theory_correct"] == False)
].copy()

print("\nCases where theory-guided prompt helped:", len(helped_cases))
print("Cases where theory-guided prompt hurt:", len(hurt_cases))
print("Cases where both prompts were wrong:", len(both_wrong_cases))

# ---------------------------------------------------------------------
# 13. Advanced Metric Upgrade: 3D Macro Binning & Continuous VAD Vector Distance
# ---------------------------------------------------------------------


def macro_map_vad(v, a, d):
    """
    Maps continuous VAD values [-1.0, 1.0] to qualitative macro categories.
    Thresholds of ±0.33 create three equal-width bins:
      Negative/Neutral/Positive for Valence,
      Low/Moderate/High for Arousal,
      Submissive/Moderate/Dominant for Dominance.
    """
    if v >= 0.33:
        val_macro = "Positive"
    elif v <= -0.33:
        val_macro = "Negative"
    else:
        val_macro = "Neutral"

    if a >= 0.33:
        aro_macro = "High"
    elif a <= -0.33:
        aro_macro = "Low"
    else:
        aro_macro = "Moderate"

    if d >= 0.33:
        dom_macro = "Dominant"
    elif d <= -0.33:
        dom_macro = "Submissive"
    else:
        dom_macro = "Moderate"

    return {"Valence": val_macro, "Arousal": aro_macro, "Dominance": dom_macro}


def get_full_macro_category(emo):
    if emo not in vad_coords:
        return "Unknown"
    coords = vad_coords[emo]
    macro = macro_map_vad(coords[0], coords[1], coords[2])
    return f"{macro['Valence']}_{macro['Arousal']}_{macro['Dominance']}"


# Full 3D Macro Bins (Valence_Arousal_Dominance)
df_sample["true_macro_3d"] = df_sample["emotion"].apply(get_full_macro_category)
df_sample["baseline_macro_3d"] = df_sample["baseline_prediction"].apply(
    get_full_macro_category
)
df_sample["theory_macro_3d"] = df_sample["theory_prediction"].apply(
    get_full_macro_category
)

baseline_macro_3d_accuracy = accuracy_score(
    df_sample["true_macro_3d"], df_sample["baseline_macro_3d"]
)
theory_macro_3d_accuracy = accuracy_score(
    df_sample["true_macro_3d"], df_sample["theory_macro_3d"]
)


def calculate_vad_error(row, pred_col):
    true_label = row["emotion"]
    pred_label = row[pred_col]
    if pred_label in ["invalid_output", "api_error"] or pd.isna(pred_label):
        return np.nan  
    if true_label in vad_coords and pred_label in vad_coords:
        return float(np.linalg.norm(vad_coords[true_label] - vad_coords[pred_label]))
    return np.nan


df_sample["baseline_vad_error"] = df_sample.apply(
    lambda r: calculate_vad_error(r, "baseline_prediction"), axis=1
)
df_sample["theory_vad_error"] = df_sample.apply(
    lambda r: calculate_vad_error(r, "theory_prediction"), axis=1
)

# Count invalid predictions excluded from VAD error
baseline_invalid = df_sample["baseline_vad_error"].isna().sum()
theory_invalid   = df_sample["theory_vad_error"].isna().sum()

baseline_mean_vad_error = df_sample["baseline_vad_error"].mean()
theory_mean_vad_error = df_sample["theory_vad_error"].mean()

baseline_incorrect_vad = df_sample[df_sample["baseline_correct"] == False][
    "baseline_vad_error"
].mean()
theory_incorrect_vad = df_sample[df_sample["theory_correct"] == False][
    "theory_vad_error"
].mean()

print("\n" + "=" * 70)
print(
    "ADVANCED METRIC UPGRADE: 3D MACRO CATEGORIES & CONTINUOUS VAD VECTOR"
    " DISTANCE"
)
print("=" * 70)
print("-" * 50)
print(f"Baseline 3D Macro-Category Accuracy:  {baseline_macro_3d_accuracy:.3f}")
print(f"Theory-Guided 3D Macro-Category Accuracy:  {theory_macro_3d_accuracy:.3f}")
print("-" * 50)
print(
    f"Overall Baseline Mean VAD Error Distance:      {baseline_mean_vad_error:.4f}"
)
print(
    f"Overall Theory-Guided Mean VAD Error Distance: {theory_mean_vad_error:.4f}"
)
print(f"Baseline predictions excluded from VAD error (invalid/missing): {baseline_invalid}")
print(f"Theory predictions excluded from VAD error (invalid/missing): {theory_invalid}")
print(
    f"Baseline Mean Error Distance (On Misses Only): {baseline_incorrect_vad:.4f}"
)
print(
    f"Theory-Guided Mean Error Distance (On Misses Only): {theory_incorrect_vad:.4f}"
)
print(
    f"VAD Error Distance Improvement (Overall):      {baseline_mean_vad_error - theory_mean_vad_error:.4f}"
)

# ---------------------------------------------------------------------
# Emotion-Specific VAD Error Distance
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("EMOTION-SPECIFIC VAD ERROR DISTANCE")
print("=" * 70)

# For 4-emotion experiment
print("\n--- 4-Emotion Experiment ---")
print(f"{'Emotion':<12} {'Baseline Mean':<15} {'Theory Mean':<15} {'Improvement':<15}")
print("-" * 60)

for emotion in target_emotions:
    mask = df_sample["emotion"] == emotion
    baseline_mean = df_sample.loc[mask, "baseline_vad_error"].mean()
    theory_mean = df_sample.loc[mask, "theory_vad_error"].mean()
    improvement = baseline_mean - theory_mean
    print(f"{emotion:<12} {baseline_mean:<15.3f} {theory_mean:<15.3f} {improvement:<15.3f}")

# Identify best and worst
vad_data = []
for emotion in target_emotions:
    mask = df_sample["emotion"] == emotion
    baseline_mean = df_sample.loc[mask, "baseline_vad_error"].mean()
    theory_mean = df_sample.loc[mask, "theory_vad_error"].mean()
    vad_data.append({
        "emotion": emotion,
        "baseline": baseline_mean,
        "theory": theory_mean,
        "improvement": baseline_mean - theory_mean
    })

best = max(vad_data, key=lambda x: x["improvement"])
worst = min(vad_data, key=lambda x: x["improvement"])

print(f"\nBest improvement: {best['emotion']} (-{best['improvement']:.3f})")
print(f"Worst improvement: {worst['emotion']} ({worst['improvement']:.3f})")

# McNemar Test: Paired Classification Accuracy Comparison
# ---------------------------------------------------------------------

baseline_correct = df_sample["baseline_correct"]
theory_correct = df_sample["theory_correct"]

# Contingency table:
#
#                 Theory Correct
#                 No        Yes
#
# Baseline No     a         b
# Baseline Yes    c         d
#

table = np.zeros((2,2), dtype=int)

for b, t in zip(baseline_correct, theory_correct):
    table[int(b), int(t)] += 1


print("\nMcNemar Contingency Table")
print(table)

result = mcnemar(
    table,
    exact=True
)

print("\nMcNemar Test")
print(f"Statistic = {result.statistic}")
print(f"p-value = {result.pvalue:.4f}")


# Effect size: improvement rate
improved = table[0,1]   # baseline wrong -> theory correct
hurt = table[1,0]       # baseline correct -> theory wrong

net_improvement = improved - hurt

print("\nDiscordant Cases")
print(f"Theory helped: {improved}")
print(f"Theory hurt: {hurt}")
print(f"Net improvement: {net_improvement}")

print("\n" + "=" * 70)
print("EMOTION-SPECIFIC MCNEMAR TESTS")
print("=" * 70)

# ---------------------------------------------------------------------
# EMOTION-SPECIFIC MCNEMAR TESTS (WITH BENJAMINI-HOCHBERG FDR CORRECTION)
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("EMOTION-SPECIFIC MCNEMAR TESTS (WITH FDR CORRECTION)")
print("=" * 70)

raw_p_values = []
targets_tested = []
summary_counts = []

for target in target_emotions:
    mask = df_sample["emotion"] == target
    b_c = df_sample.loc[mask, "baseline_correct"]
    t_c = df_sample.loc[mask, "theory_correct"]
    
    table_class = np.zeros((2,2), dtype=int)
    for b, t in zip(b_c, t_c):
        table_class[int(b), int(t)] += 1
        
    res = mcnemar(table_class, exact=True)
    raw_p_values.append(res.pvalue)
    targets_tested.append(target)
    summary_counts.append((table_class[0,1], table_class[1,0]))

# Apply Benjamini-Hochberg Correction
reject, pvals_corrected, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')

for i, target in enumerate(targets_tested):
    improved, hurt = summary_counts[i]
    print(f"--- {target.upper()} ---")
    print(f"Theory Helped: {improved} | Theory Hurt: {hurt} | Net: {improved - hurt}")
    print(f"Raw p-value: {raw_p_values[i]:.4f} | FDR Adjusted p-value: {pvals_corrected[i]:.4f}")
    if reject[i]:
        direction = "IMPROVED" if improved > hurt else "DEGRADED"
        print(f"* Statistically significant {direction} (Adjusted) *")
    else:
        print("No statistically significant change after FDR correction.")
    print("")
    
    
#  Mann‑Whitney U
# ---------------------------------------------------------------------    
    
baseline_err_misses = df_sample.loc[df_sample["baseline_correct"]==False, "baseline_vad_error"].dropna()
theory_err_misses = df_sample.loc[df_sample["theory_correct"]==False, "theory_vad_error"].dropna()

u_stat, mw_p = mannwhitneyu(baseline_err_misses, theory_err_misses, alternative='two-sided')
print("\nMann-Whitney U on VAD error for incorrect predictions only:")
print(f"Baseline median error (misses): {baseline_err_misses.median():.4f}")
print(f"Theory median error (misses): {theory_err_misses.median():.4f}")
print(f"U = {u_stat:.1f}, p = {mw_p:.4f}")


# ---------------------------------------------------------------------
# Keep VAD distance analysis separately
# ---------------------------------------------------------------------

diff = (
    df_sample["baseline_vad_error"] -
    df_sample["theory_vad_error"]
).dropna()   # <-- drop NaN so paired test runs on valid rows only

t_stat, p_val = stats.ttest_1samp(diff, 0)

print("\nVAD Distance Paired t-test")
print(f"t = {t_stat:.3f}")
print(f"p = {p_val:.4f}")

# 95% confidence interval
sem = stats.sem(diff)
ci = stats.t.interval(
    confidence=0.95,
    df=len(diff)-1,
    loc=diff.mean(),
    scale=sem
)

print(f"95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

# Wilcoxon signed-rank
w = stats.wilcoxon(diff)

print(f"Wilcoxon W = {w.statistic:.1f}")
print(f"Wilcoxon p = {w.pvalue:.4f}")

print("\n" + "=" * 70)
print("BOOTSTRAP CONFIDENCE INTERVALS (Accuracy Difference)")
print("=" * 70)

n_iterations = 1000
n_size = len(df_sample)
diffs = []

for i in range(n_iterations):
    indices = np.random.choice(n_size, size=n_size, replace=True)
    y_true_boot = df_sample["emotion"].iloc[indices]
    y_base_boot = df_sample["baseline_prediction"].iloc[indices]
    y_theo_boot = df_sample["theory_prediction"].iloc[indices]
    
    base_acc = accuracy_score(y_true_boot, y_base_boot)
    theo_acc = accuracy_score(y_true_boot, y_theo_boot)
    diffs.append(theo_acc - base_acc)

lower_bound = np.percentile(diffs, 2.5)
upper_bound = np.percentile(diffs, 97.5)
mean_diff = np.mean(diffs)

print(f"Mean Bootstrapped Difference: {mean_diff:.4f}")
print(f"95% Confidence Interval for Difference: [{lower_bound:.4f}, {upper_bound:.4f}]")
if lower_bound > 0:
    print("Conclusion: Theory-guided prompt is SIGNIFICANTLY BETTER at 95% confidence.")
elif upper_bound < 0:
    print("Conclusion: Baseline prompt is SIGNIFICANTLY BETTER at 95% confidence.")
else:
    print("Conclusion: No significant difference in overall accuracy (0 is within CI).")
    
# ---------------------------------------------------------------------
# 14. Save Outputs
# ---------------------------------------------------------------------
df_sample.to_csv("emotion_llm_full_results.csv", index=False)

baseline_confusion_df.to_csv("baseline_confusion_matrix.csv")
theory_confusion_df.to_csv("theory_guided_confusion_matrix.csv")

baseline_errors = df_sample[
    df_sample["emotion"] != df_sample["baseline_prediction"]
].copy()

theory_errors = df_sample[
    df_sample["emotion"] != df_sample["theory_prediction"]
].copy()

baseline_errors.to_csv("baseline_error_analysis.csv", index=False)
theory_errors.to_csv("theory_guided_error_analysis.csv", index=False)

helped_cases.to_csv("theory_helped_cases.csv", index=False)
hurt_cases.to_csv("theory_hurt_cases.csv", index=False)
both_wrong_cases.to_csv("both_wrong_cases.csv", index=False)

print("\nFiles saved:")
print("- emotion_llm_full_results.csv")
print("- baseline_confusion_matrix.csv")
print("- theory_guided_confusion_matrix.csv")
print("- baseline_error_analysis.csv")
print("- theory_guided_error_analysis.csv")
print("- theory_helped_cases.csv")
print("- theory_hurt_cases.csv")
print("- both_wrong_cases.csv")

print("\nPipeline complete.") 