import os
import pandas as pd
import numpy as np

os.makedirs("d:/Capstone_Project/data/metadata", exist_ok=True)
os.makedirs("d:/Capstone_Project/data/dummy_daic", exist_ok=True)

meta = []
for i in range(16):
    pid = 300 + i
    split = "train" if i < 12 else "dev"
    folder = f"d:/Capstone_Project/data/dummy_daic/{pid}_P"
    os.makedirs(folder, exist_ok=True)
    meta.append({
        "participant_id": pid,
        "split": split,
        "gender": 1,
        "phq8_binary": i % 2,
        "phq8_score": 12 if i % 2 else 5,
        "audio_path": f"dummy_daic/{pid}_P/audio.wav",
        "transcript_path": f"dummy_daic/{pid}_P/trans.csv",
        "covarep_path": f"dummy_daic/{pid}_P/cov.csv",
        "formant_path": f"dummy_daic/{pid}_P/form.csv"
    })
    
    pd.DataFrame(np.random.randn(300, 74)).to_csv(f"{folder}/cov.csv", header=False, index=False)
    pd.DataFrame(np.random.randn(300, 5)).to_csv(f"{folder}/form.csv", header=False, index=False)
    
    trans_df = pd.DataFrame([
        {"start_time": 1.0, "stop_time": 3.0, "speaker": "Ellie", "value": "hello participant"},
        {"start_time": 3.5, "stop_time": 6.0, "speaker": "Participant", "value": f"hello i am participant {pid} feeling {'sad' if i%2 else 'good'} today"},
        {"start_time": 6.5, "stop_time": 9.0, "speaker": "Ellie", "value": "how are you doing"},
        {"start_time": 9.5, "stop_time": 12.0, "speaker": "Participant", "value": "i am answering all questions clearly and concisely"}
    ])
    trans_df.to_csv(f"{folder}/trans.csv", sep="\t", index=False)

pd.DataFrame(meta).to_csv("d:/Capstone_Project/data/metadata/metadata.csv", index=False)
print("Updated dummy dataset metadata with train and dev splits and valid transcripts.")
