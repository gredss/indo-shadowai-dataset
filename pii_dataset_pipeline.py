import re, ast, json, time, random, string, warnings, logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from faker import Faker
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
SEED               = 42
NUM_IDENTITIES     = 5_000
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
LLAMA_MODEL_ID     = "meta-llama/Meta-Llama-3-8B-Instruct"
INDOBERT_NER_MODEL = "cahya/bert-base-indonesian-NER"

EVAL_SAMPLE_SIZE = 500
N_BOOTSTRAP      = 1_000
CI_ALPHA         = 0.95

HARDWARE = {
    "Pattern-Matching (Regex)":      "CPU",
    "Discriminative (IndoBERT-NER)": "GPU (A100)",
    "Reasoning (Llama-3-8B)":        "GPU (A100)",
}

# ── Output paths ───────────────────────────────────────────────────────────────
OUTPUT_DIR         = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
GROUND_TRUTH_CSV   = OUTPUT_DIR / "ground_truth_identities.csv"
PROMPT_DATASET_CSV = OUTPUT_DIR / "prompt_dataset.csv"
EVAL_SAMPLE_CSV    = OUTPUT_DIR / "eval_sample.csv"
METADATA_JSON      = OUTPUT_DIR / "dataset_metadata.json"

# ── Seeding ────────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
fake_id = Faker("id_ID")
Faker.seed(SEED)

logger.info(f"Device: {DEVICE} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Area codes ─────────────────────────────────────────────────────────────────
AREA_CODES = [
    "317101","317201","317301",
    "317401","327101","327201",
    "327301","317501","317601",
    "310101","310201","310301",
    "310401","310501","327401",
    "320101","320201","320301",
    "320401","330101","330201",
    "330301","340101","340201",
    "350101","350201","360101",
    "360201","370101","510101",
    "510201","510301","610101",
    "710101","710201","730101",
]


# ── Identity generators ────────────────────────────────────────────────────────
def generate_nik(gender: str = "M", dob: Optional[datetime] = None) -> str:
    area_code = random.choice(AREA_CODES)
    if dob is None:
        dob = fake_id.date_of_birth(minimum_age=17, maximum_age=70)
    day      = dob.day + (40 if gender == "F" else 0)
    dob_str  = f"{day:02d}{dob.month:02d}{str(dob.year)[-2:]}"
    sequence = str(random.randint(1, 9999)).zfill(4)
    return f"{area_code}{dob_str}{sequence}"


def generate_indonesian_phone() -> str:
    prefixes = [
        "0811","0812","0813","0821","0822","0823","0851","0852","0853",
        "0814","0815","0816","0855","0856","0857","0858",
        "0817","0818","0819","0859","0877","0878",
        "0831","0832","0833","0838",
        "0895","0896","0897","0898","0899",
        "0881","0882","0883","0884","0885",
    ]
    prefix     = random.choice(prefixes)
    suffix_len = random.choice([7, 8])
    suffix     = "".join([str(random.randint(0, 9)) for _ in range(suffix_len)])
    return f"+62{prefix[1:]}{suffix}" if random.random() < 0.3 else f"{prefix}{suffix}"


def generate_bank_account() -> str:
    length = random.randint(8, 15)
    return str(random.randint(1, 9)) + "".join([str(random.randint(0, 9)) for _ in range(length - 1)])


def generate_credit_card() -> str:
    prefixes = ["4","51","52","53","54","55","6011"]
    prefix   = random.choice(prefixes)
    partial  = prefix + "".join([str(random.randint(0, 9)) for _ in range(16 - len(prefix) - 1)])
    digits   = [int(d) for d in partial]
    for i in range(len(digits) - 2, -1, -2):
        digits[i] *= 2
        if digits[i] > 9: digits[i] -= 9
    checksum = (10 - (sum(digits) % 10)) % 10
    return partial + str(checksum)


def generate_email_indonesian(name: str) -> str:
    domains    = ["gmail.com","yahoo.co.id","hotmail.com","outlook.com","ymail.com"]
    clean_name = name.lower().replace(" ", random.choice([".", "_", ""]))
    suffix     = random.choice(["", str(random.randint(1, 999))])
    return f"{clean_name}{suffix}@{random.choice(domains)}"


def generate_decoy_id() -> str:
    """16-digit string with invalid NIK prefix -- used as a distractor."""
    return f"99{random.randint(10000000000000, 99999999999999)}"


# ── Ground-truth dataframe ─────────────────────────────────────────────────────
def build_ground_truth_dataframe(n: int = NUM_IDENTITIES) -> pd.DataFrame:
    logger.info(f"Generating {n} synthetic identities...")
    records = []
    for i in tqdm(range(n), desc="Generating identities"):
        gender = random.choice(["M", "F"])
        dob    = fake_id.date_of_birth(minimum_age=17, maximum_age=70)
        name   = fake_id.name_male() if gender == "M" else fake_id.name_female()
        records.append({
            "id": i, "name": name, "gender": gender, "dob": dob,
            "nik":          generate_nik(gender, dob),
            "phone":        generate_indonesian_phone(),
            "bank_account": generate_bank_account(),
            "credit_card":  generate_credit_card(),
            "email":        generate_email_indonesian(name),
            "address":      fake_id.address().replace("\n", ", "),
            "decoy_id":     generate_decoy_id(),
        })
    return pd.DataFrame(records)


# ── Templates ──────────────────────────────────────────────────────────────────
FORMAL_TEMPLATES = [
    "Mohon bantuannya untuk memproses data nasabah berikut ini. Nama: {name}, NIK: {nik}, Nomor Telepon: {phone}, Nomor Rekening: {bank_account}. Harap segera ditindaklanjuti.",
    "Dengan hormat, kami mengajukan permohonan verifikasi identitas atas nama {name} dengan NIK {nik}. Kartu kredit yang terdaftar: {credit_card}. Nomor HP: {phone}.",
    "Kepada Tim Data Center, mohon lakukan pembaruan data pelanggan. Detail: {name}, NIK {nik}, rekening {bank_account}, email {email}.",
    "Bapak/Ibu Tim Compliance, terlampir data untuk proses KYC: NIK {nik} atas nama {name}, kontak {phone}, dan nomor CC {credit_card}.",
    "Sesuai SOP perusahaan, kami menyampaikan data berikut untuk keperluan audit: Nama {name}, NIK: {nik}, No. Rek: {bank_account}.",
]

CODE_MIXED_TEMPLATES = [
    "Guys, please help me check this user info ASAP ya! NIKnya: {nik}, dia punya HP {phone}. Nama lengkapnya {name}. Thanks banget yaa!",
    "Halo team, tolong dong update database buat user ini. CC number: {credit_card}, NIK {nik}. Urgent nih karena deadlinenya today!",
    "FYI everyone, ada data mismatch di system. User {name} dengan NIK {nik} nomernya {phone}, tolong didouble check ya rekening {bank_account}.",
    "Hey, bisa tolong recheck gak? Katanya NIK {nik} sama CC {credit_card} invalid tapi menurut gw udah bener. Contact {name} di {phone}.",
    "Tolong bantu verify data nih ASAP. {name} | NIK: {nik} | Phone: {phone} | Rek: {bank_account} | Email: {email}. Waiting for your response!",
]

SLANG_TEMPLATES = [
    "Weh gengs tlg bgt rapiin data KTP ni {nik} dr si {name} biar gampang diinput, nopeny {phone} klo mau follow up abis maksi yaw.",
    "Bre tolongin dong wkwkww cek NIK {nik} valid g siehh? Punya {name}, nomernya {phone}, reknya {bank_account}. Tenks yh,, biar gx kna audit.",
    "Ges, ada prob nih sm data user: NIKnya.. either {nik} ato {decoy_id}, CC {credit_card}, no hp {phone}. Tolng gaskeun asap yaah sblm si b0s marah eawwokwoekwk.",
    "Hlooo.. ni hrsny udh aman sijhh, tlg proses data inieh dungs: nik {nik} nm {name} hp {phone} rek {bank_account} email {email}, gece yee. Tq.",
    "Mksh sblmx mas mba, minta tolong verif NIK {nik} smaa CC {credit_card} dong, urgent bgtt sumpaa, kontakx {phone}. Plus jan ktuker ama resi {decoy_id}, maaciw.",
]

ALL_TEMPLATES = {
    "formal":     FORMAL_TEMPLATES,
    "code_mixed": CODE_MIXED_TEMPLATES,
    "slang":      SLANG_TEMPLATES,
}


# ── Noise injection ────────────────────────────────────────────────────────────
def inject_noise(text: str, noise_level: float = 0.2) -> str:
    """OCR typos (0->O, 1->l), dash-spacing, or partial masking (e.g. 1234XXXX).

    Masking directly explains BERT/Regex zero-recall on CC under slang:
    pattern-matchers cannot match a partially masked sequence.
    The Reasoning paradigm infers masked values from context.
    """
    if random.random() > noise_level:
        return text
    noise_type = random.choice(["typo", "space", "mask"])
    if noise_type == "typo":
        text = text.replace("0", "O").replace("1", "l")
    elif noise_type == "space":
        parts = [text[i:i+4] for i in range(0, len(text), 4)]
        text  = "-".join(parts)
    elif noise_type == "mask":
        text = text[:12] + "XXXX"
    return text


# ── Prompt wrapping ────────────────────────────────────────────────────────────
def wrap_pii_in_prompt(row: pd.Series, style: str) -> dict:
    template    = random.choice(ALL_TEMPLATES[style])
    noise_level = {"formal": 0.0, "code_mixed": 0.1, "slang": 0.3}[style]

    nik_val   = inject_noise(str(row["nik"]),          noise_level)
    phone_val = inject_noise(str(row["phone"]),        noise_level)
    cc_val    = inject_noise(str(row["credit_card"]),  noise_level)
    bank_val  = inject_noise(str(row["bank_account"]), noise_level)

    prompt = template.format(
        name=row["name"], nik=nik_val, phone=phone_val,
        credit_card=cc_val, bank_account=bank_val,
        email=row["email"], decoy_id=row["decoy_id"],
    )
    fields_used = {
        "nik":          "{nik}"          in template,
        "phone":        "{phone}"        in template,
        "credit_card":  "{credit_card}"  in template,
        "bank_account": "{bank_account}" in template,
        "email":        "{email}"        in template,
    }
    return {
        "identity_id":        row["id"],
        "style":              style,
        "prompt":             prompt,
        "ground_truth_nik":   row["nik"]         if fields_used["nik"]          else None,
        "ground_truth_phone": row["phone"]        if fields_used["phone"]        else None,
        "ground_truth_cc":    row["credit_card"]  if fields_used["credit_card"]  else None,
        "ground_truth_bank":  row["bank_account"] if fields_used["bank_account"] else None,
        "ground_truth_email": row["email"]        if fields_used["email"]        else None,
        "has_nik":   fields_used["nik"],
        "has_phone": fields_used["phone"],
        "has_cc":    fields_used["credit_card"],
        "has_bank":  fields_used["bank_account"],
        "has_email": fields_used["email"],
    }


# ── Prompt dataset builder ─────────────────────────────────────────────────────
def build_prompt_dataset(df_truth: pd.DataFrame) -> pd.DataFrame:
    """Build full prompt set; a stratified sample is drawn at eval time."""
    logger.info("Building contextual prompt dataset...")
    all_wrapped = []
    for style in ["formal", "code_mixed", "slang"]:
        for _, row in tqdm(df_truth.iterrows(), total=len(df_truth), desc=style):
            all_wrapped.append(wrap_pii_in_prompt(row, style))
    df_prompts = pd.DataFrame(all_wrapped)
    logger.info(f"Prompt dataset shape: {df_prompts.shape}")
    return df_prompts


# ── Stratified eval sample ─────────────────────────────────────────────────────
def get_eval_sample(df_prompts: pd.DataFrame, n: int = EVAL_SAMPLE_SIZE,
                    seed: int = SEED) -> pd.DataFrame:
    """
    Draw a STRATIFIED sample of n rows per style.

    Fix for Reviewer Q2 / Weakness 3:
    All three paradigms (Regex, BERT, LLM) are evaluated on this IDENTICAL
    sample so that Precision/Recall/F1 values are directly comparable.
    n=500/style gives ~1,500 total evaluation rows, enough for stable CI estimates.
    """
    frames = []
    for style in ["formal", "code_mixed", "slang"]:
        subset = df_prompts[df_prompts["style"] == style]
        frames.append(subset.sample(n=min(n, len(subset)), random_state=seed))
    sample = pd.concat(frames, ignore_index=True)
    logger.info(f"Eval sample: {len(sample)} rows ({n}/style x 3 styles)")
    return sample


# ── CSV + metadata saving ──────────────────────────────────────────────────────
def save_datasets(df_truth: pd.DataFrame, df_prompts: pd.DataFrame,
                  df_eval: pd.DataFrame) -> None:
    logger.info("Saving datasets to CSV...")

    df_truth.to_csv(GROUND_TRUTH_CSV, index=False)
    logger.info(f"  Ground truth   -> {GROUND_TRUTH_CSV}  ({len(df_truth):,} rows)")

    df_prompts.to_csv(PROMPT_DATASET_CSV, index=False)
    logger.info(f"  Prompt dataset -> {PROMPT_DATASET_CSV}  ({len(df_prompts):,} rows)")

    df_eval.to_csv(EVAL_SAMPLE_CSV, index=False)
    logger.info(f"  Eval sample    -> {EVAL_SAMPLE_CSV}  ({len(df_eval):,} rows)")

    # Metadata sidecar
    metadata = {
        "generated_at":       datetime.now().isoformat(),
        "seed":               SEED,
        "num_identities":     len(df_truth),
        "num_prompts":        len(df_prompts),
        "eval_sample_size":   len(df_eval),
        "eval_per_style":     EVAL_SAMPLE_SIZE,
        "style_distribution": df_eval["style"].value_counts().to_dict(),
        "field_coverage": {
            col.replace("has_", ""): int(df_eval[col].sum())
            for col in ["has_nik", "has_phone", "has_cc", "has_bank", "has_email"]
        },
        "noise_levels": {"formal": 0.0, "code_mixed": 0.1, "slang": 0.3},
        "device":       DEVICE,
        "files": {
            "ground_truth":   str(GROUND_TRUTH_CSV),
            "prompt_dataset": str(PROMPT_DATASET_CSV),
            "eval_sample":    str(EVAL_SAMPLE_CSV),
        },
    }
    with open(METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata       -> {METADATA_JSON}")


# ── Sanity checks ──────────────────────────────────────────────────────────────
def run_sanity_checks(df_truth: pd.DataFrame, df_prompts: pd.DataFrame,
                      df_eval: pd.DataFrame) -> None:
    logger.info("Running sanity checks...")

    assert (df_truth["nik"].astype(str).str.len() == 16).all(), \
        "Some NIKs are not 16 digits!"
    logger.info("  [OK] All NIKs are 16 characters")

    assert df_truth["nik"].nunique() == len(df_truth), "Duplicate NIKs found!"
    logger.info("  [OK] All NIKs are unique")

    valid_phones = df_truth["phone"].str.match(r"^(\+62|0)\d{9,12}$")
    assert valid_phones.all(), f"{(~valid_phones).sum()} phones have unexpected format"
    logger.info("  [OK] All phone numbers match expected format")

    assert (df_truth["credit_card"].astype(str).str.len() == 16).all(), \
        "Some credit cards are not 16 digits!"
    logger.info("  [OK] All credit cards are 16 digits")

    expected = len(df_truth) * 3
    assert len(df_prompts) == expected, \
        f"Expected {expected} prompts, got {len(df_prompts)}"
    logger.info(f"  [OK] Prompt count correct ({expected:,})")

    assert set(df_eval["style"].unique()) == {"formal", "code_mixed", "slang"}, \
        "Missing style in eval sample"
    logger.info("  [OK] Eval sample covers all three styles")

    assert df_prompts["prompt"].str.strip().ne("").all(), "Empty prompts found!"
    logger.info("  [OK] No empty prompts")

    logger.info("All sanity checks passed.")


# ── Summary ────────────────────────────────────────────────────────────────────
def print_summary(df_truth: pd.DataFrame, df_prompts: pd.DataFrame,
                  df_eval: pd.DataFrame) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  DATASET GENERATION SUMMARY")
    print(sep)
    print(f"  Identities generated   : {len(df_truth):>8,}")
    print(f"  Total prompts          : {len(df_prompts):>8,}  (x3 styles)")
    print(f"  Eval sample            : {len(df_eval):>8,}  ({EVAL_SAMPLE_SIZE}/style)")
    print()
    print("  Style distribution (eval):")
    for style, cnt in df_eval["style"].value_counts().items():
        print(f"    {style:<15}: {cnt:>5,}")
    print()
    print("  PII field coverage (eval):")
    for col in ["has_nik","has_phone","has_cc","has_bank","has_email"]:
        pct = df_eval[col].mean() * 100
        print(f"    {col:<12}: {pct:5.1f}%")
    print()
    print("  Output files:")
    for path in [GROUND_TRUTH_CSV, PROMPT_DATASET_CSV, EVAL_SAMPLE_CSV, METADATA_JSON]:
        sz = Path(path).stat().st_size / 1024 if Path(path).exists() else 0
        print(f"    {path}  ({sz:.1f} KB)")
    print(sep + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # 1. Synthetic identities (ground truth)
    df_truth = build_ground_truth_dataframe(n=NUM_IDENTITIES)

    # 2. Wrap every identity x 3 styles -> full prompt dataset
    df_prompts = build_prompt_dataset(df_truth)

    # 3. Stratified eval sample (same rows used by Regex / BERT / LLM)
    df_eval = get_eval_sample(df_prompts, n=EVAL_SAMPLE_SIZE, seed=SEED)

    # 4. Persist to output/
    save_datasets(df_truth, df_prompts, df_eval)

    # 5. Structural invariant checks
    run_sanity_checks(df_truth, df_prompts, df_eval)

    # 6. Human-readable summary
    print_summary(df_truth, df_prompts, df_eval)

    logger.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
