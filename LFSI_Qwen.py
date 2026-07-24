import os
import random
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from EDI_value import calculate_effective_space, get_feature_data_with_cache


DATA_FOLDER = "./Data/Seen"
LLM_PATH = "./Qwen2.5-1.5B-Instruct"   # Pre-download the LLM weight files
CACHE_PATH = "./dataset_cache/MotorB_feature_cache.pkl"

INIT_FEATURES = 10
MAX_ITER = 50
MIN_FEATURES = 5
MAX_FEATURES = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_GROUPS = {

    "time_domain": [
        "mean",
        "variance",
        "std_dev",
        "rms",
        "max_value",
        "min_value",
        "absolute_mean",
        "peak_value",
        "peak_to_peak",
        "energy",
        "integrated_signal",
        "root_sum_squares",
        "log_rms",
        "bi_segment_mav",
        "tri_segment_mav"
    ],

    "impulsive": [
        "kurtosis",
        "skewness",
        "waveform_index",
        "impulse_factor",
        "crest_factor",
        "clearance_factor",
        "kurtosis_index",
        "peak_index",
        "pulse_index",
        "mean_deviation_ratio",
        "root_mean_fourth",
        "square_root_amplitude",
        "peak_count"
    ],

    "statistical_moments": [
        "fifth_statistical_moment",
        "sixth_statistical_moment",
        "kth_central_moment"
    ],

    "entropy_nonlinear": [
        "shannon_entropy",
        "log_energy_entropy",
        "sample_entropy",
        "correlation_dimension"
    ],

    "dynamic_variation": [
        "slope_sign_change",
        "zero_crossing_rate",
        "mav_slope",
        "delta_rms",
        "average_amplitude_change"
    ],

    "frequency_domain": [
        "frequency_mean",
        "frequency_variance",
        "frequency_skewness",
        "frequency_kurtosis",
        "gravity_frequency",
        "frequency_std",
        "frequency_rms",
        "dominant_frequency",
        "fundamental_amplitude",
        "second_harmonic_amplitude",
        "third_harmonic_amplitude",
        "spectral_spread",
        "spectral_entropy",
        "total_spectral_power",
        "first_spectral_moment",
        "second_spectral_moment",
        "third_spectral_moment",
        "fourth_spectral_moment",
        "low_band_spectral_energy",
        "mid_band_spectral_energy",
        "high_band_spectral_energy",
        "harmonic_ratio",
        "mean_square_frequency",
        "spectral_flux",
        "spectral_flatness",
        "frequency_distortion",
        "spectral_rolloff"
    ],

    "time_frequency": [
        "stft_mean",
        "stft_variance",
        "stft_skewness",
        "stft_kurtosis",
        "stft_entropy",
        "wavelet_energy",
        "wavelet_entropy",
        "cwt_mean",
        "cwt_variance",
        "cwt_skewness",
        "cwt_kurtosis",
        "hilbert_envelope_rms"
    ],

    "power_spectrum": [
        "power_spectral_density_mean",
        "power_spectral_density_max"
    ]
}


#  Call LLM
def call_llm(prompt, max_new_tokens=300):
    """Call LLM and return JSON string"""
    messages = [
        {"role": "system", "content": "You are an expert in vibration signal fault diagnosis and feature engineering."},
        {"role": "user", "content": prompt}
    ]

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_PATH, trust_remote_code=True).to(DEVICE)
    model.eval()

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the last JSON block
    matches = re.findall(r'\{.*?\}', response, re.DOTALL)
    if matches:
        return matches[-1]
    return "{}"


def get_feature_group(feature_name):

    for group, feats in FEATURE_GROUPS.items():

        if feature_name in feats:
            return group

    return "unknown"


def summarize_feature_groups(features):

    summary = {}

    for f in features:

        group = get_feature_group(f)

        summary.setdefault(
            group,
            []
        ).append(f)

    return summary


def random_initialize_features(available_features):
    """Randomly initialize the feature subset"""
    if len(available_features) <= INIT_FEATURES:
        return available_features.copy()

    selected = []
    for group, feats in FEATURE_GROUPS.items():
        candidates = [f for f in feats if f in available_features]
        if candidates:
            selected.append(random.choice(candidates))

    remaining = [f for f in available_features if f not in selected]
    need_more = INIT_FEATURES - len(selected)

    if need_more > 0 and remaining:
        selected.extend(random.sample(remaining, min(need_more, len(remaining))))

    while len(selected) < INIT_FEATURES and remaining:
        f = random.choice(remaining)
        selected.append(f)
        remaining.remove(f)

    print(f"\nRandom Initialization Success: {len(selected)} features")
    return selected


def get_best_feature_combination(folder_path: str):

    selected_csvs = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    )

    features_df, fault_labels, _ = get_feature_data_with_cache(
        folder_path=folder_path,
        selected_csvs=selected_csvs,
        cache_path=CACHE_PATH,
        sample_length=2048,
        step=512,
        max_samples_per_csv=300
    )

    available_features = list(features_df.columns)

    # LLM initialization
    current_attrs = random_initialize_features(
        available_features
    )

    best_attrs = current_attrs.copy()
    best_score = -1e9

    # Top-K elite archive
    elite_archive = []

    # Taboo feature set
    tabu_features = {}

    # Critical feature records
    critical_features = {}
    last_added = []
    last_removed = []
    history = []

    no_improve_count = 0
    CONVERGE_THRESHOLD = 8

    ROLLBACK_THRESHOLD = 0.03
    TABU_ROUNDS = 5

    print(current_attrs)

    for iteration in range(MAX_ITER):
        # Decrease remaining rounds for tabu features
        for feat in list(tabu_features.keys()):

            tabu_features[feat] -= 1

            if tabu_features[feat] <= 0:
                del tabu_features[feat]

        # EDI evaluation
        total_space, I_joint, avg_Q, per_attr_space = (
            calculate_effective_space(
                features_df,
                fault_labels,
                current_attrs
            )
        )

        if total_space > best_score + 1e-8:
            best_score = total_space
            best_attrs = current_attrs.copy()
            no_improve_count = 0

        else:
            no_improve_count += 1

        if no_improve_count >= CONVERGE_THRESHOLD:
            print(f"\n Convergence detected after {iteration+1} iterations")
            break

        if best_score > 0:
            drop_ratio = (best_score - total_space) / (best_score + 1e-8)
        else:
            drop_ratio = 0

        if iteration > 3 and drop_ratio > ROLLBACK_THRESHOLD:

            # Add recently added features to tabu list
            for feat in last_added:
                tabu_features[feat] = TABU_ROUNDS

            # Mark removed features as critical features
            for feat in last_removed:
                critical_features[feat] = (
                        critical_features.get(feat, 0) + 1
                )

            # Randomly recover from top5 elite solutions
            if len(elite_archive) > 0:

                elite_score, elite_attrs = random.choice(elite_archive)
                current_attrs = elite_attrs.copy()

            else:
                current_attrs = best_attrs.copy()

            continue

        current_attrs = [
            f for f in current_attrs
            if per_attr_space.get(f, 0.0) > 1e-8
        ]

        # Maintain feature quantity constraints
        if len(current_attrs) < MIN_FEATURES:

            remain = [
                f for f in available_features
                if f not in current_attrs
            ]

            current_attrs.extend(
                random.sample(
                    remain,
                    min(
                        MIN_FEATURES - len(current_attrs),
                        len(remain)
                    )
                )
            )

        if len(current_attrs) > MAX_FEATURES:

            current_attrs = sorted(
                current_attrs,
                key=lambda x:
                per_attr_space.get(x, 0),
                reverse=True
            )[:MAX_FEATURES]

        elite_archive.append((float(total_space), current_attrs.copy()))
        elite_archive = sorted(
            elite_archive,
            key=lambda x: x[0],
            reverse=True
        )[:7]

        # Record optimization history
        history.append(
            {
                "iter": iteration + 1,
                "edi": round(float(total_space), 6),
                "features": current_attrs.copy()
            }
        )

        recent_history = history[-5:]

        print(
            f"Iteration {iteration+1:02d}   "
            f"EDI={total_space:.6f}   "
            f"Best={best_score:.6f}   "
            f"N={len(current_attrs)}"
        )

        # Build candidate pool
        remaining_features = [
            f for f in available_features
            if (f not in current_attrs and f not in tabu_features)
        ]

        high_score_candidates = sorted(
            remaining_features,
            key=lambda x:
            critical_features.get(x, 0),
            reverse=True
        )

        candidate_pool = (high_score_candidates[:10])

        random_candidates = [

            f for f in remaining_features
            if f not in candidate_pool
        ]

        candidate_pool += random.sample(
            random_candidates,
            min(10, len(random_candidates))
        )

        # Determine current search state
        search_state = "exploit"

        if len(history) >= 5:

            old_edi = history[-5]["edi"]
            new_edi = history[-1]["edi"]

            improvement = (
                new_edi - old_edi
            ) / (abs(old_edi) + 1e-8)

            if improvement < 0.001:
                search_state = "explore"

            elif random.random() < 0.25:
                search_state = "explore"

        # Assemble history text
        history_text = ""

        for h in recent_history:

            history_text += f"Iter {h['iter']}: "f"EDI={h['edi']}\n"

        # Feature contribution scores for current subset
        feature_scores = {
            f: round(float(per_attr_space.get(f, 0)), 6)
            for f in current_attrs
        }

        current_group_info = summarize_feature_groups(current_attrs)

        # Prompt
        prompt = f"""
        You are an expert vibration signal diagnostician and a feature subset reasoning agent for rotating machinery fault diagnosis.       
        Your goal is NOT simply removing low-score features.
        You should reason using:
        1. fault diagnosis knowledge
        2. feature complementarity
        3. optimization history
        4. EDI feedback
        
        Current subset:        
        {current_attrs}
        
        Current EDI:        
        {total_space:.6f}
        
        Per-feature EDI:        
        {json.dumps(feature_scores, indent=2)}
        
        Recent optimization history:        
        {history_text}
        
        Current tabu features:
        {list(tabu_features.keys())}
        
        Critical features:
        {list(critical_features.keys())}
        
        Candidate features:        
        {candidate_pool}
                
        Current feature distribution:
        {json.dumps(current_group_info, indent=2)}
        
        Current search mode:        
        {search_state}
        
        Rules:
        1. exploit:
           add/remove <=1 feature       
        2. explore:
           add/remove <=3 features        
        3. Do NOT recommend tabu features.        
        4. Avoid removing critical features.        
        5. Preserve diversity among:
           - time_domain
           - frequency_domain
           - time_frequency
           - entropy_nonlinear        
        6. Prefer under-represented groups.        
        7. Maximize future EDI.
        
        Output JSON only:        
        {{
        "strategy":"explore or exploit",
        "reason":"short reason",
        "add":["feature1"],
        "remove":["feature2"]
        }}
        """

        try:

            llm_output = call_llm(prompt)

            action = json.loads(llm_output)

            strategy = action.get("strategy", search_state)

            reason = action.get("reason", "")

            to_add = [
                f for f in action.get("add", [])
                if (f in available_features and f not in current_attrs)
            ]

            to_remove = [
                f for f in action.get("remove", [])
                if f in current_attrs
            ]

            last_added = to_add.copy()
            last_removed = to_remove.copy()

            if strategy == "exploit":

                to_add = to_add[:1]
                to_remove = to_remove[:1]

            else:

                to_add = to_add[:3]
                to_remove = to_remove[:3]

            # print("\nLLM Reasoning")
            # print("Strategy:", strategy)
            # # print("Reason:", reason)         # Uncomment this line if you want to view the reasoning from LLM
            # print("ADD:", to_add)
            # print("REMOVE:", to_remove)

            # Execute feature adjustment actions
            for f in to_remove:

                if f in current_attrs:
                    current_attrs.remove(f)

            for f in to_add:

                if f not in current_attrs and len(current_attrs) < MAX_FEATURES:
                    current_attrs.append(f)

        except Exception as e:

            print(f"LLM Parse Failed: {e}")

            continue

    print("Feature Optimization Finished")
    print(f"Best EDI = {best_score:.6f}")
    print(f"Feature Number = {len(best_attrs)}")

    return best_attrs
