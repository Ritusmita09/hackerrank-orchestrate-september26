"""
Usage logger for Buy or Wait? challenge.

Tracks token usage and generates usage report.
"""
from typing import Dict, List
from pathlib import Path


class UsageLogger:
    """Track token usage and costs."""

    def __init__(self):
        self.model_usage = {}  # model_name -> {calls, input_tokens, output_tokens}
        self.total_requests = 0

    def log_call(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        """Log a single API call."""
        key = f"{provider}/{model}"

        if key not in self.model_usage:
            self.model_usage[key] = {
                'provider': provider,
                'model': model,
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
            }

        self.model_usage[key]['calls'] += 1
        self.model_usage[key]['input_tokens'] += input_tokens
        self.model_usage[key]['output_tokens'] += output_tokens

    def set_total_requests(self, count: int):
        """Set total number of requests processed."""
        self.total_requests = count

    def generate_report(self, output_path: str = "code/evaluation/usage_report.md"):
        """Generate usage report markdown file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        total_calls = sum(m['calls'] for m in self.model_usage.values())
        total_input = sum(m['input_tokens'] for m in self.model_usage.values())
        total_output = sum(m['output_tokens'] for m in self.model_usage.values())
        total_tokens = total_input + total_output

        avg_tokens_per_request = total_tokens / self.total_requests if self.total_requests > 0 else 0

        # Estimate costs (approximate rates)
        total_cost = self._estimate_cost()
        cost_per_request = total_cost / self.total_requests if self.total_requests > 0 else 0

        with open(output_path, 'w') as f:
            f.write("# Token Usage Report\n\n")
            f.write("## Summary\n\n")

            # List providers
            providers = set(m['provider'] for m in self.model_usage.values())
            f.write(f"- **Provider(s)**: {', '.join(sorted(providers))}\n")

            # List models
            models = [m['model'] for m in self.model_usage.values()]
            f.write(f"- **Model(s)**: {', '.join(models)}\n")

            f.write(f"- **Total Requests Processed**: {self.total_requests}\n")
            f.write(f"- **Total API Calls**: {total_calls}\n")
            f.write(f"- **Total Input Tokens**: {total_input:,}\n")
            f.write(f"- **Total Output Tokens**: {total_output:,}\n")
            f.write(f"- **Total Tokens**: {total_tokens:,}\n")
            f.write(f"- **Average Tokens per Request**: {avg_tokens_per_request:,.1f}\n")
            f.write(f"- **Estimated Total Cost**: ${total_cost:.4f}\n")
            f.write(f"- **Estimated Cost per Request**: ${cost_per_request:.4f}\n\n")

            if self.model_usage:
                f.write("## Per-Model Breakdown\n\n")
                f.write("| Provider | Model | Calls | Input Tokens | Output Tokens | Total Tokens | Est. Cost |\n")
                f.write("|----------|-------|-------|--------------|---------------|--------------|----------|\n")

                for key in sorted(self.model_usage.keys()):
                    usage = self.model_usage[key]
                    provider = usage['provider']
                    model = usage['model']
                    calls = usage['calls']
                    input_tok = usage['input_tokens']
                    output_tok = usage['output_tokens']
                    total_tok = input_tok + output_tok
                    cost = self._estimate_model_cost(provider, model, input_tok, output_tok)

                    f.write(f"| {provider} | {model} | {calls} | {input_tok:,} | {output_tok:,} | {total_tok:,} | ${cost:.4f} |\n")

            f.write("\n## Notes\n\n")
            f.write("- Cost estimates based on published pricing as of September 2026\n")
            f.write("- Actual costs may vary based on caching, batching, and pricing changes\n")
            f.write("- Token usage includes only image extraction and message interpretation API calls\n")
            f.write("- All financial calculations and decision logic are deterministic Python code\n")

        print(f"Usage report written to {output_path}")

    def _estimate_cost(self) -> float:
        """Estimate total cost based on token usage."""
        total_cost = 0.0

        for usage in self.model_usage.values():
            provider = usage['provider']
            model = usage['model']
            input_tokens = usage['input_tokens']
            output_tokens = usage['output_tokens']

            total_cost += self._estimate_model_cost(provider, model, input_tokens, output_tokens)

        return total_cost

    def _estimate_model_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a specific model."""
        # Anthropic pricing (approximate, September 2026)
        if provider.lower() == 'anthropic':
            if 'opus' in model.lower():
                # Claude 3 Opus: $15 / 1M input, $75 / 1M output
                return (input_tokens / 1_000_000 * 15.0) + (output_tokens / 1_000_000 * 75.0)
            elif 'sonnet' in model.lower():
                # Claude 3.5 Sonnet: $3 / 1M input, $15 / 1M output
                return (input_tokens / 1_000_000 * 3.0) + (output_tokens / 1_000_000 * 15.0)
            elif 'haiku' in model.lower():
                # Claude 3 Haiku: $0.25 / 1M input, $1.25 / 1M output
                return (input_tokens / 1_000_000 * 0.25) + (output_tokens / 1_000_000 * 1.25)

        # OpenAI pricing (approximate)
        elif provider.lower() == 'openai':
            if 'gpt-4' in model.lower():
                # GPT-4: $30 / 1M input, $60 / 1M output
                return (input_tokens / 1_000_000 * 30.0) + (output_tokens / 1_000_000 * 60.0)
            elif 'gpt-3.5' in model.lower():
                # GPT-3.5: $0.50 / 1M input, $1.50 / 1M output
                return (input_tokens / 1_000_000 * 0.50) + (output_tokens / 1_000_000 * 1.50)

        # Default estimate
        return (input_tokens / 1_000_000 * 3.0) + (output_tokens / 1_000_000 * 15.0)

    def get_summary(self) -> Dict:
        """Get usage summary as dictionary."""
        total_input = sum(m['input_tokens'] for m in self.model_usage.values())
        total_output = sum(m['output_tokens'] for m in self.model_usage.values())
        total_tokens = total_input + total_output
        total_cost = self._estimate_cost()

        return {
            'total_requests': self.total_requests,
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_tokens': total_tokens,
            'estimated_cost': total_cost,
            'models': self.model_usage,
        }
