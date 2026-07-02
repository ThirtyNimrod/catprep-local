import re
from pathlib import Path
from collections import defaultdict
from core.token_tracker import TOKEN_LOG_FILE

def parse_token_log(path: Path | str = TOKEN_LOG_FILE) -> dict:
    path = Path(path)
    stats = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    
    if not path.exists():
        return dict(stats)
        
    usage_pattern = re.compile(r"\[(.*?)\] Prompt: (\d+), Completion: (\d+), Total: (\d+)")
    no_usage_pattern = re.compile(r"\[(.*?)\] No usage found")
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            usage_match = usage_pattern.search(line)
            if usage_match:
                caller = usage_match.group(1)
                prompt = int(usage_match.group(2))
                completion = int(usage_match.group(3))
                total = int(usage_match.group(4))
                
                stats[caller]["calls"] += 1
                stats[caller]["prompt_tokens"] += prompt
                stats[caller]["completion_tokens"] += completion
                stats[caller]["total_tokens"] += total
                continue
                
            no_usage_match = no_usage_pattern.search(line)
            if no_usage_match:
                caller = no_usage_match.group(1)
                stats["_untracked"]["calls"] += 1
                
    return dict(stats)

def format_summary_table(stats: dict) -> str:
    if not stats:
        return "No token usage recorded."
        
    lines = []
    lines.append(f"{'Caller':<25} | {'Calls':<6} | {'Prompt':<8} | {'Completion':<10} | {'Total':<8}")
    lines.append("-" * 65)
    
    sorted_callers = sorted(
        [k for k in stats.keys() if k != "_untracked"],
        key=lambda k: stats[k]["total_tokens"],
        reverse=True
    )
    
    if "_untracked" in stats:
        sorted_callers.append("_untracked")
        
    for caller in sorted_callers:
        data = stats[caller]
        lines.append(
            f"{caller:<25} | {data['calls']:<6} | {data['prompt_tokens']:<8} | "
            f"{data['completion_tokens']:<10} | {data['total_tokens']:<8}"
        )
        
    return "\n".join(lines)
