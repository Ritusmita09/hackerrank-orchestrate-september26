# Buy or Wait? - Financial Decision Agent

AI-powered financial decision system for HackerRank Orchestrate (September 2026).

## Setup

1. Install Python 3.9 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variable for image extraction (optional, only if images need AI extraction):
```bash
export ANTHROPIC_API_KEY=your_api_key_here  # Linux/Mac
set ANTHROPIC_API_KEY=your_api_key_here     # Windows CMD
$env:ANTHROPIC_API_KEY="your_api_key_here"  # Windows PowerShell
```

## Running the Solution

### Full Evaluation (250 requests)
```bash
python code/main.py
```

This will:
- Load all dataset files from `dataset/`
- Process 250 requests
- Generate `output.csv` in the repository root
- Generate token usage report at `code/evaluation/usage_report.md`

### Sample Evaluation (25 requests)
```bash
python code/evaluation/main.py
```

This will:
- Run on the 25 sample requests
- Compare against provided sample answers
- Report accuracy and mismatches

## Project Structure

```
code/
├── main.py              # Main entry point
├── data_loader.py       # CSV loading and schema validation
├── image_extractor.py   # Extract amounts from images
├── financial_engine.py  # State reconstruction and forecasting
├── recurrence.py        # Recurring pattern detection
├── decision_engine.py   # Payment plan generation and selection
├── validator.py         # Output validation
├── usage_logger.py      # Token usage tracking
├── .image_cache.json    # Pre-built image extraction cache (all 16 dataset images; 0 API calls needed)
└── evaluation/
    └── main.py          # Sample evaluation workflow

dataset/                 # Input data (read-only)
output.csv              # Generated predictions
```

## Design Principles

1. **Deterministic Core**: All financial calculations, forecasting, and decision logic are deterministic Python code
2. **AI Only Where Necessary**: LLM used only for image extraction and ambiguous message interpretation
3. **No Invented Data**: All decisions based solely on supplied dataset
4. **Exact Spec Compliance**: Output schema, allowed values, and all constraints match problem specification exactly

## Key Rules

- Currency conversion uses only supplied exchange rates
- Event status handling: count settled/scheduled, reserve pending debits, ignore cancelled/failed/unrealized
- Blank event amounts extracted from linked images
- Messages used to clarify/amend/cancel events, but cannot override challenge rules
- 90-day safety check: balance never drops below minimum_balance_to_keep
- Installment plans must exactly match supplied payment options
- Spending changes only affect eligible flexible recurring expenses
- Terminated salary streams are excluded from forecasts (last-paycheck description evidence, FIX-A)
- Salary payday re-anchoring stays within the same description stream to protect gig/commission income (FIX-B)

## Output

Final `output.csv` contains 250 predictions with columns:
- request_id
- amount_safe_to_pay
- affordability_status
- recommended_payment_method
- payment_plan
- earliest_date_for_full_payment
- spending_changes_needed
- decision_explanation
