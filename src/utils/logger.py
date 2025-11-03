from datetime import datetime
import json
from config import LOGS_PATH


def save_training_results(results, log):
    results['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(f"{LOGS_PATH}/{log}", 'a') as file:
        json.dump(results, file)
        file.write("\n\n")

