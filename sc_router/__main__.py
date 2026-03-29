"""CLI for SC-Router.

Usage:
    python -m sc_router classify "query text"
    python -m sc_router profiles [--model MODEL]
    python -m sc_router profiles reset [--model MODEL] [--level LEVEL]
"""

import argparse
import sys

from .catalog import Tool, ToolCatalog
from .classifier import classify_query


def _cmd_classify(args):
    """Classify a query and print the SC level."""
    # Use an empty catalog if none provided (features still compute)
    catalog = ToolCatalog()
    result = classify_query(args.query, catalog)
    level = result['level']
    confidence = result['confidence']
    phase = result['phase']

    strategies = {0: 'direct', 1: 'pipeline', 2: 'search', 3: 'agent'}
    strategy = strategies.get(level, 'unknown')

    print(f"SC({level})  confidence={confidence}  strategy={strategy}  phase={phase}")

    if args.verbose:
        features = result.get('evidence', {}).get('features', {})
        if features:
            print("\nFeatures:")
            for k, v in sorted(features.items()):
                print(f"  {k}: {v}")

        patterns = result.get('evidence', {}).get('patterns', {})
        if patterns:
            print(f"\nPattern shortcut: {patterns.get('shortcut_available', False)}")
            if patterns.get('shortcut_available'):
                print(f"  source: {patterns.get('shortcut_source')}")
                print(f"  confidence: {patterns.get('shortcut_confidence')}")


def _cmd_profiles(args):
    """List or reset model quality profiles."""
    try:
        from .profiles import ProfileManager
    except ImportError:
        print("Error: llm-ekg not installed. Run: pip install sc-router[ekg]",
              file=sys.stderr)
        sys.exit(1)

    pm = ProfileManager()

    if args.action == 'reset':
        model = getattr(args, 'model', None)
        level = getattr(args, 'level', None)
        pm.reset(model=model, sc_level=level)
        pm.save()
        print("Profiles reset.")
        return

    # List profiles
    profiles = pm.all_profiles()

    if args.model:
        profiles = [p for p in profiles if p['model'] == args.model]

    if not profiles:
        print("No profiles found. Record responses with ProfileManager.record().")
        return

    # Header
    print(f"{'Model':<25} {'SC':>3} {'Score':>6} {'Verdict':<10} "
          f"{'H.Risk':>6} {'N':>5} {'Trend':<8}")
    print("-" * 72)

    for p in profiles:
        print(f"{p['model']:<25} {p['sc_level']:>3} "
              f"{p['global_score_100']:>6} {p['verdict']:<10} "
              f"{p['hallucination_risk']:>6.3f} {p['n_responses']:>5} "
              f"{p['trend']:<8}")


def main():
    parser = argparse.ArgumentParser(
        prog='sc-router',
        description='SC-Router: AI routing based on Selector Complexity',
    )
    subparsers = parser.add_subparsers(dest='command')

    # classify
    p_classify = subparsers.add_parser('classify', help='Classify a query')
    p_classify.add_argument('query', help='The query text to classify')
    p_classify.add_argument('-v', '--verbose', action='store_true',
                            help='Show detailed features and patterns')

    # profiles
    p_profiles = subparsers.add_parser('profiles', help='Manage model profiles')
    p_profiles.add_argument('action', nargs='?', default='list',
                            choices=['list', 'reset'],
                            help='Action: list (default) or reset')
    p_profiles.add_argument('--model', help='Filter by model name')
    p_profiles.add_argument('--level', type=int,
                            help='Filter/reset by SC level (0-3)')

    args = parser.parse_args()

    if args.command == 'classify':
        _cmd_classify(args)
    elif args.command == 'profiles':
        _cmd_profiles(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
