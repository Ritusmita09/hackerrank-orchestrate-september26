"""
Image extractor for Buy or Wait? challenge.

Extracts missing amounts from images using vision API or cached values.
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict
import anthropic
from PIL import Image


class ImageExtractor:
    """Extract amounts from financial document images."""

    def __init__(self, dataset_dir: str = "dataset", cache_file: str = "code/.image_cache.json"):
        self.dataset_dir = Path(dataset_dir)
        self.media_dir = self.dataset_dir / "media" / "images"
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = None
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _load_cache(self) -> Dict:
        """Load cached extraction results."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save extraction results to cache."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def extract_amount(self, image_id: str, event_description: str, currency: str) -> float:
        """
        Extract amount from image.

        Args:
            image_id: Image identifier (e.g., 'image_01')
            event_description: Event description for context
            currency: Expected currency

        Returns:
            Extracted amount as float
        """
        # Check cache first
        cache_key = f"{image_id}_{currency}"
        if cache_key in self.cache:
            print(f"Using cached extraction for {image_id}: {self.cache[cache_key]} {currency}")
            return float(self.cache[cache_key])

        # Load image
        image_path = self.media_dir / f"{image_id}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"Extracting amount from {image_id}...")

        # Try vision API if available
        if self.client:
            amount = self._extract_with_vision_api(image_path, event_description, currency)
        else:
            raise RuntimeError(
                f"No ANTHROPIC_API_KEY found. Cannot extract amount from {image_id}. "
                "Please set ANTHROPIC_API_KEY environment variable."
            )

        # Cache result
        self.cache[cache_key] = amount
        self._save_cache()

        return amount

    def _extract_with_vision_api(self, image_path: Path, event_description: str, currency: str) -> float:
        """Extract amount using Claude vision API."""
        import base64

        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')

        # Determine image type
        image_type = "image/png"

        prompt = f"""Extract the primary payment amount from this financial document.

Event description: {event_description}
Expected currency: {currency}

Instructions:
1. Look for the net payment amount, total amount, or salary amount
2. For payslips, extract the net pay (after deductions)
3. For receipts/invoices, extract the total amount paid
4. Return ONLY the numeric value without currency symbols, commas, or formatting
5. Use decimal point (.) for fractional amounts

Return only the number."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )

        self.call_count += 1
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        # Parse response
        extracted_text = response.content[0].text.strip()

        # Clean and convert to float
        cleaned = extracted_text.replace(',', '').replace(' ', '')
        try:
            amount = float(cleaned)
            print(f"  Extracted: {amount} {currency}")
            return amount
        except ValueError:
            raise ValueError(f"Could not parse amount from vision response: {extracted_text}")

    def get_usage_stats(self) -> Dict:
        """Return API usage statistics."""
        return {
            'calls': self.call_count,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
        }
