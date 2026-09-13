"""
Data loader for Buy or Wait? challenge.

Loads and validates all CSV files from dataset/.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


def parse_bool_column(series: pd.Series) -> pd.Series:
    """Parse a boolean column that pandas may have read as native bool or as string.

    pd.read_csv already converts lowercase true/false to native bool dtype; mapping
    those values against string keys silently produces NaN, which is truthy. Handle
    both representations explicitly.
    """
    return series.map({
        True: True, False: False,
        'true': True, 'false': False,
        'True': True, 'False': False,
    })


class DataLoader:
    """Load and validate all dataset files."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)

    def load_all(self) -> Dict:
        """Load all dataset files and return as dictionary."""
        print("Loading dataset files...")

        data = {
            'profiles': self.load_profiles(),
            'events': self.load_events(),
            'requests': self.load_requests(),
            'sample_requests': self.load_sample_requests(),
            'payment_options': self.load_payment_options(),
            'exchange_rates': self.load_exchange_rates(),
            'messages': self.load_messages(),
            'images': self.load_images(),
        }

        print(f"Loaded {len(data['profiles'])} profiles")
        print(f"Loaded {len(data['events'])} events")
        print(f"Loaded {len(data['requests'])} requests")
        print(f"Loaded {len(data['sample_requests'])} sample requests")
        print(f"Loaded {len(data['payment_options'])} payment options")
        print(f"Loaded {len(data['exchange_rates'])} exchange rates")
        print(f"Loaded {len(data['messages'])} messages")
        print(f"Loaded {len(data['images'])} images")

        return data

    def load_profiles(self) -> pd.DataFrame:
        """Load financial profiles."""
        df = pd.read_csv(self.dataset_dir / "financial_profiles.csv")

        # Parse pipe-separated fields
        df['financial_priorities'] = df['financial_priorities'].fillna('').str.split('|')
        df['expense_categories_to_protect'] = df['expense_categories_to_protect'].fillna('').str.split('|')
        df['expense_categories_user_is_willing_to_reduce'] = df['expense_categories_user_is_willing_to_reduce'].fillna('').str.split('|')
        df['expense_categories_user_is_willing_to_stop'] = df['expense_categories_user_is_willing_to_stop'].fillna('').str.split('|')
        df['payment_methods_user_will_consider'] = df['payment_methods_user_will_consider'].fillna('').str.split('|')

        # Handle empty strings after split
        for col in ['financial_priorities', 'expense_categories_to_protect',
                    'expense_categories_user_is_willing_to_reduce',
                    'expense_categories_user_is_willing_to_stop',
                    'payment_methods_user_will_consider']:
            df[col] = df[col].apply(lambda x: [item for item in x if item])

        # Default to full_payment only if empty
        df['payment_methods_user_will_consider'] = df['payment_methods_user_will_consider'].apply(
            lambda x: x if x else ['full_payment']
        )

        return df

    def load_events(self) -> pd.DataFrame:
        """Load financial events."""
        df = pd.read_csv(self.dataset_dir / "financial_events.csv")

        # Parse dates
        df['event_date'] = pd.to_datetime(df['event_date'])
        df['settlement_date'] = pd.to_datetime(df['settlement_date'])

        # Handle blank amounts (will be filled from images)
        df['amount'] = df['amount'].replace('', None)

        return df

    def load_requests(self) -> pd.DataFrame:
        """Load evaluation requests."""
        df = pd.read_csv(self.dataset_dir / "requests.csv")

        # Parse dates
        df['request_date'] = pd.to_datetime(df['request_date'])
        df['desired_completion_date'] = pd.to_datetime(df['desired_completion_date'])
        df['allows_partial_payment'] = parse_bool_column(df['allows_partial_payment'])

        return df

    def load_sample_requests(self) -> pd.DataFrame:
        """Load sample requests with answers."""
        df = pd.read_csv(self.dataset_dir / "sample_requests.csv")

        # Parse dates
        df['request_date'] = pd.to_datetime(df['request_date'])
        df['desired_completion_date'] = pd.to_datetime(df['desired_completion_date'])
        df['allows_partial_payment'] = parse_bool_column(df['allows_partial_payment'])

        # Parse earliest_date_for_full_payment (may be empty)
        df['earliest_date_for_full_payment'] = pd.to_datetime(
            df['earliest_date_for_full_payment'], errors='coerce'
        )

        return df

    def load_payment_options(self) -> pd.DataFrame:
        """Load request payment options."""
        df = pd.read_csv(self.dataset_dir / "request_payment_options.csv")

        # Parse dates
        df['first_payment_date'] = pd.to_datetime(df['first_payment_date'])

        return df

    def load_exchange_rates(self) -> pd.DataFrame:
        """Load exchange rates."""
        df = pd.read_csv(self.dataset_dir / "exchange_rates.csv")

        # Parse dates
        df['rate_date'] = pd.to_datetime(df['rate_date'])

        return df

    def load_messages(self) -> pd.DataFrame:
        """Load messages."""
        df = pd.read_csv(self.dataset_dir / "messages.csv")

        # Parse timestamp and convert to timezone-naive (remove UTC timezone)
        df['sent_at'] = pd.to_datetime(df['sent_at']).dt.tz_localize(None)

        return df

    def load_images(self) -> pd.DataFrame:
        """Load image metadata."""
        df = pd.read_csv(self.dataset_dir / "images.csv")
        return df

    def build_lookup_structures(self, data: Dict) -> Dict:
        """Build efficient lookup structures for related data."""
        lookups = {}

        # Profile by user_id
        lookups['profile_by_user'] = {
            row['user_id']: row
            for _, row in data['profiles'].iterrows()
        }

        # Events by user_id
        lookups['events_by_user'] = data['events'].groupby('user_id').apply(
            lambda x: x.to_dict('records')
        ).to_dict()

        # Payment options by request_id
        lookups['options_by_request'] = data['payment_options'].groupby('request_id').apply(
            lambda x: x.to_dict('records')
        ).to_dict()

        # Messages by user_id
        lookups['messages_by_user'] = data['messages'].groupby('user_id').apply(
            lambda x: x.to_dict('records')
        ).to_dict()

        # Messages by request_id
        lookups['messages_by_request'] = data['messages'][
            data['messages']['request_id'].notna()
        ].groupby('request_id').apply(
            lambda x: x.to_dict('records')
        ).to_dict()

        # Images by related_event_id
        lookups['images_by_event'] = {
            row['related_event_id']: row
            for _, row in data['images'].iterrows()
        }

        # Exchange rates lookup: (from_cur, to_cur) -> rates_df
        lookups['exchange_rates'] = data['exchange_rates']

        return lookups


def get_exchange_rate(rate_date: datetime, from_currency: str, to_currency: str,
                      rates_df: pd.DataFrame) -> float:
    """
    Get exchange rate for a specific date and currency pair.

    Uses the closest available rate on or before the given date.
    """
    if from_currency == to_currency:
        return 1.0

    # Find rates for this currency pair
    pair_rates = rates_df[
        (rates_df['from_currency'] == from_currency) &
        (rates_df['to_currency'] == to_currency) &
        (rates_df['rate_date'] <= rate_date)
    ].sort_values('rate_date', ascending=False)

    if len(pair_rates) > 0:
        return pair_rates.iloc[0]['rate']

    # Try inverse rate
    inverse_rates = rates_df[
        (rates_df['from_currency'] == to_currency) &
        (rates_df['to_currency'] == from_currency) &
        (rates_df['rate_date'] <= rate_date)
    ].sort_values('rate_date', ascending=False)

    if len(inverse_rates) > 0:
        return 1.0 / inverse_rates.iloc[0]['rate']

    raise ValueError(f"No exchange rate found for {from_currency} -> {to_currency} on {rate_date}")


def convert_to_home_currency(amount: float, from_currency: str, to_currency: str,
                             settlement_date: datetime, rates_df: pd.DataFrame) -> float:
    """Convert amount to home currency using dated exchange rates."""
    if from_currency == to_currency:
        return amount

    rate = get_exchange_rate(settlement_date, from_currency, to_currency, rates_df)
    return amount * rate
