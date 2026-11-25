"""Send test-set customers to the API to generate metrics and a prediction log.

python scripts/simulate_traffic.py --n 300            # normal traffic
python scripts/simulate_traffic.py --n 300 --drift    # shifted traffic -> drift alert
"""

import argparse
import time

import httpx
import pandas as pd

from churn.config import TEST_PATH
from churn.schema import FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--drift", action="store_true", help="simulate a shifted population")
    parser.add_argument("--delay", type=float, default=0.02)
    args = parser.parse_args()

    df = pd.read_csv(TEST_PATH)[FEATURES].sample(n=args.n, replace=True, random_state=0)
    if args.drift:
        # new pricing + a marketing push to new, month-to-month customers
        df["MonthlyCharges"] = (df["MonthlyCharges"] * 1.4).round(2)
        df["tenure"] = (df["tenure"] * 0.3).astype(int)
        df["Contract"] = "Month-to-month"
        df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)

    churned = 0
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for record in df.to_dict(orient="records"):
            r = client.post("/predict", json=record)
            r.raise_for_status()
            churned += r.json()["churn"]
            time.sleep(args.delay)
    print(f"Sent {args.n} requests - predicted churn rate {churned / args.n:.1%}")


if __name__ == "__main__":
    main()
