import argparse
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.token_summary import parse_token_log, format_summary_table
from core.token_tracker import TOKEN_LOG_FILE

def main():
    parser = argparse.ArgumentParser(description="Summarize LLM token usage")
    parser.add_argument("--log-file", type=str, default=str(TOKEN_LOG_FILE), help="Path to token_usage.log")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()
    
    stats = parse_token_log(args.log_file)
    
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(format_summary_table(stats))

if __name__ == "__main__":
    main()
