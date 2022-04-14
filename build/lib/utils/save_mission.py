"""Utility file for saving and loading a mission."""
import os
from datetime import datetime
import pickle


def get_mission_hash(mission_name="modelsim"):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H:%M:%S")
    return f"{mission_name}_{dt_string}"


def save_mission(mission_hash, rob, model, env=None, simulator=None):
    filepath = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/{mission_hash}.pkl")
    with open(filepath, "wb") as fh:
        pickle.dump({
            "robot": rob,
            "model": model,
            "environment": env,
            "simulator": simulator,
        }, fh)


def load_mission(mission_hash):
    filepath = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/{mission_hash}.pkl")
    with open(filepath, "rb") as fh:
        data = pickle.load(fh)
        return data["robot"], data["model"], data["environment"], data["simulator"]
