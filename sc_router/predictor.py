"""Threshold-based decision tree for SC level prediction.

No ML dependencies — uses threshold splits on the 17 features.
Training data: ~40 labeled queries with known SC levels.
"""

from typing import Dict, List, Optional, Tuple

from .features import FEATURE_NAMES


class SCRouterPredictor:
    """Predict SC level (0-3) from query features using threshold splits.

    - For each feature, find the best threshold that separates SC levels.
    - At prediction time, each split votes (weighted by accuracy).
    - Fallback: nearest-mean classification.
    """

    def __init__(self):
        self._splits: List[Tuple[str, float, int, int, float]] = []
        # (feature_name, threshold, level_below, level_above, accuracy)
        self._level_means: Dict[int, Dict[str, float]] = {}
        self._fitted = False

    def fit(self, training_data: List[Tuple[Dict[str, float], int]]) -> 'SCRouterPredictor':
        """Fit from labeled (features_dict, sc_level) pairs.

        For each feature, tries every midpoint threshold between sorted unique
        values and picks the split with highest classification accuracy.
        Keeps splits with accuracy > 0.6.
        """
        if not training_data:
            return self

        # Group by level
        by_level: Dict[int, List[Dict[str, float]]] = {}
        for features, level in training_data:
            by_level.setdefault(level, []).append(features)

        # Compute per-level feature means (for fallback)
        self._level_means = {}
        for level, samples in by_level.items():
            means = {}
            for fname in FEATURE_NAMES:
                vals = [s.get(fname, 0.0) for s in samples]
                means[fname] = sum(vals) / max(len(vals), 1)
            self._level_means[level] = means

        # Find best threshold splits
        self._splits = []
        all_levels = sorted(by_level.keys())

        for fname in FEATURE_NAMES:
            # Collect (value, level) pairs
            pairs = []
            for features, level in training_data:
                pairs.append((features.get(fname, 0.0), level))
            pairs.sort(key=lambda x: x[0])

            # Try midpoint thresholds
            values = sorted(set(v for v, _ in pairs))
            if len(values) < 2:
                continue

            best_acc = 0.0
            best_split = None

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2.0

                # For each pair of levels, try this threshold
                for l_below in all_levels:
                    for l_above in all_levels:
                        if l_below == l_above:
                            continue
                        correct = 0
                        total = 0
                        for val, level in pairs:
                            if level == l_below and val <= threshold:
                                correct += 1
                                total += 1
                            elif level == l_above and val > threshold:
                                correct += 1
                                total += 1
                            elif level in (l_below, l_above):
                                total += 1
                        if total > 0:
                            acc = correct / total
                            if acc > best_acc:
                                best_acc = acc
                                best_split = (fname, threshold, l_below, l_above, acc)

            if best_split and best_acc > 0.6:
                self._splits.append(best_split)

        # Sort by accuracy descending
        self._splits.sort(key=lambda x: -x[4])
        self._fitted = True
        return self

    def predict(self, features: Dict[str, float]) -> Dict:
        """Predict SC level from features.

        Returns dict with 'level', 'confidence', 'method'.
        """
        if not self._fitted:
            return {'level': 2, 'confidence': 'low', 'method': 'unfitted'}

        # Weighted voting from splits
        votes: Dict[int, float] = {}
        for fname, threshold, l_below, l_above, acc in self._splits:
            val = features.get(fname, 0.0)
            predicted = l_below if val <= threshold else l_above
            votes[predicted] = votes.get(predicted, 0.0) + acc

        if votes:
            total_weight = sum(votes.values())
            winner = max(votes, key=votes.get)
            winner_weight = votes[winner]
            ratio = winner_weight / total_weight if total_weight > 0 else 0

            if ratio > 0.7:
                confidence = 'high'
            elif ratio > 0.5:
                confidence = 'medium'
            else:
                confidence = 'low'

            return {'level': winner, 'confidence': confidence, 'method': 'threshold_voting'}

        # Fallback: nearest-mean
        return self._nearest_mean(features)

    def _nearest_mean(self, features: Dict[str, float]) -> Dict:
        """Classify by L2 distance to level means."""
        best_level = 0
        best_dist = float('inf')

        for level, means in self._level_means.items():
            dist = 0.0
            for fname in FEATURE_NAMES:
                diff = features.get(fname, 0.0) - means.get(fname, 0.0)
                dist += diff * diff
            if dist < best_dist:
                best_dist = dist
                best_level = level

        return {'level': best_level, 'confidence': 'low', 'method': 'nearest_mean'}

    def fit_from_examples(self) -> 'SCRouterPredictor':
        """Auto-fit from built-in training examples.

        ~40 queries covering SC(0) through SC(3), with pre-computed feature
        signatures.
        """
        training_data = _get_training_data()
        return self.fit(training_data)


def _get_training_data() -> List[Tuple[Dict[str, float], int]]:
    """Built-in training dataset of labeled query feature signatures.

    Contains ~40 synthetic examples (parametric feature vectors) plus
    ~20 real examples (features extracted from actual queries against
    a 10-tool catalog). The '_source' field is metadata only and is
    ignored by the predictor.
    """
    data = []

    # ===================================================================
    # SYNTHETIC EXAMPLES (40 total, 10 per SC level)
    # Parametric feature vectors covering the feature space.
    # ===================================================================

    # --- SC(0): Single tool, obvious dispatch ---
    for i in range(10):
        nc = 0.0 + (i % 3) * 0.5
        nt = 1.0 + (i % 2)
        data.append(({
            'num_constraints': nc,
            'num_candidate_tools': nt,
            'max_dependency_depth': 1.0,
            'avg_dependency_depth': 1.0,
            'tool_coverage': 0.05 + i * 0.01,
            'tool_overlap': 0.0,
            'query_expansion': 1.0,
            'constraint_tool_ratio': nc / max(nt, 1),
            'single_tool_tasks': 1.0,
            'two_tool_tasks': 0.0,
            'multi_tool_tasks': 0.0,
            'avg_task_breadth': nt,
            'max_task_breadth': nt,
            'tool_clusters': 1.0,
            'max_cluster_size': nt,
            'spectral_gap': 0.0,
            'avg_tool_specificity': 0.8 + i * 0.01,
            '_source': 'synthetic',
        }, 0))

    # --- SC(1): Decomposable pipeline ---
    for i in range(10):
        nsub = 2.0 + (i % 2)
        nt = 2.0 + i * 0.3
        data.append(({
            'num_constraints': 1.0 + (i % 3),
            'num_candidate_tools': nt,
            'max_dependency_depth': nsub,
            'avg_dependency_depth': nsub * 0.7,
            'tool_coverage': 0.1 + i * 0.02,
            'tool_overlap': 0.1 + i * 0.03,
            'query_expansion': nsub,
            'constraint_tool_ratio': (1.0 + i % 3) / max(nt, 1),
            'single_tool_tasks': nsub * 0.6,
            'two_tool_tasks': nsub * 0.3,
            'multi_tool_tasks': 0.0,
            'avg_task_breadth': 1.3 + i * 0.05,
            'max_task_breadth': 2.0,
            'tool_clusters': 1.0 + (i % 2),
            'max_cluster_size': nt * 0.7,
            'spectral_gap': 0.3 + i * 0.05,
            'avg_tool_specificity': 0.6 + i * 0.02,
            '_source': 'synthetic',
        }, 1))

    # --- SC(2): Ambiguous/complex, search needed ---
    for i in range(10):
        nsub = 3.0 + i * 0.5
        nt = 4.0 + i * 0.5
        nc = 2.0 + i * 0.5
        data.append(({
            'num_constraints': nc,
            'num_candidate_tools': nt,
            'max_dependency_depth': 2.0 + i * 0.3,
            'avg_dependency_depth': 1.5 + i * 0.2,
            'tool_coverage': 0.2 + i * 0.03,
            'tool_overlap': 0.3 + i * 0.04,
            'query_expansion': nsub * 0.8,
            'constraint_tool_ratio': nc / max(nt, 1),
            'single_tool_tasks': 1.0,
            'two_tool_tasks': nsub * 0.4,
            'multi_tool_tasks': nsub * 0.3,
            'avg_task_breadth': 2.5 + i * 0.2,
            'max_task_breadth': 4.0 + i * 0.3,
            'tool_clusters': 2.0 + (i % 3),
            'max_cluster_size': nt * 0.5,
            'spectral_gap': 0.5 + i * 0.04,
            'avg_tool_specificity': 0.4 + i * 0.03,
            '_source': 'synthetic',
        }, 2))

    # --- SC(3): Globally entangled, full agent ---
    for i in range(10):
        nsub = 4.0 + i
        nt = 6.0 + i
        nc = 3.0 + i
        data.append(({
            'num_constraints': nc,
            'num_candidate_tools': nt,
            'max_dependency_depth': 3.0 + i * 0.5,
            'avg_dependency_depth': 2.5 + i * 0.4,
            'tool_coverage': 0.4 + i * 0.04,
            'tool_overlap': 0.5 + i * 0.04,
            'query_expansion': nsub,
            'constraint_tool_ratio': nc / max(nt, 1),
            'single_tool_tasks': 0.0,
            'two_tool_tasks': 1.0,
            'multi_tool_tasks': nsub * 0.6,
            'avg_task_breadth': 3.5 + i * 0.3,
            'max_task_breadth': 6.0 + i * 0.5,
            'tool_clusters': 1.0,
            'max_cluster_size': nt,
            'spectral_gap': 0.8 + i * 0.02,
            'avg_tool_specificity': 0.2 + i * 0.02,
            '_source': 'synthetic',
        }, 3))

    # ===================================================================
    # REAL EXAMPLES (20 total, 5 per SC level)
    # Features extracted from actual queries against the standard
    # 10-tool catalog. Manually classified by SC level.
    # ===================================================================
    data.extend(_get_real_training_data())

    return data


def _get_real_training_data() -> List[Tuple[Dict[str, float], int]]:
    """Extract features from real queries against a reference catalog.

    Builds the standard 10-tool catalog and runs extract_query_features
    on each query to get ground-truth feature vectors. This ensures
    the training data matches what the predictor sees at inference.
    """
    from .catalog import Tool, ToolCatalog
    from .features import extract_query_features

    catalog = ToolCatalog()
    tools = [
        ("weather", "Get weather forecast for a location",
         {"location"}, {"weather_data"},
         {"weather", "forecast", "temperature", "climate"}),
        ("calculator", "Perform arithmetic calculations",
         {"expression"}, {"number"},
         {"math", "calculate", "arithmetic", "number"}),
        ("search", "Search the web for information",
         {"query"}, {"search_results"},
         {"search", "web", "find", "lookup", "information"}),
        ("summarizer", "Summarize text content",
         {"text", "search_results"}, {"summary"},
         {"summarize", "summary", "condense", "text"}),
        ("translator", "Translate text between languages",
         {"text", "summary"}, {"translated_text"},
         {"translate", "language", "translation"}),
        ("flight_search", "Search for flights between cities",
         {"origin", "destination", "date"}, {"flight_list"},
         {"flight", "flights", "travel", "book", "airline"}),
        ("hotel_search", "Search for hotels in a city",
         {"city", "date"}, {"hotel_list"},
         {"hotel", "hotels", "accommodation", "travel", "book"}),
        ("restaurant_search", "Find restaurants in a location",
         {"location"}, {"restaurant_list"},
         {"restaurant", "restaurants", "food", "dining"}),
        ("budget_optimizer", "Optimize spending within a budget constraint",
         {"flight_list", "hotel_list", "restaurant_list", "number"},
         {"optimized_plan"},
         {"budget", "optimize", "cost", "plan", "spending"}),
        ("sentiment_analyzer", "Analyze sentiment of text and social media data",
         {"text", "search_results"}, {"sentiment_data"},
         {"sentiment", "analyze", "social", "opinion", "trend"}),
    ]
    for name, desc, inp, out, tags in tools:
        catalog.register(Tool(
            name=name, description=desc,
            input_types=inp, output_types=out, capability_tags=tags,
        ))

    # (query, sc_level) — manually classified
    real_queries = [
        # SC(0): Direct dispatch, single tool
        ("What is the weather in Madrid?", 0),
        ("Calculate 15 * 37", 0),
        ("How is the climate in Tokyo?", 0),
        ("Search for Python tutorials", 0),
        ("Translate hello to French", 0),

        # SC(1): Decomposable pipeline
        ("Search for AI news, then summarize the key findings", 1),
        ("Find flights to Paris, then translate the options to Spanish", 1),
        ("Search for machine learning papers and summarize the abstracts", 1),
        ("Look up the weather forecast and summarize it", 1),
        ("Search for restaurant reviews and translate them to English", 1),

        # SC(2): Constrained search, multiple tools with constraints
        ("Find flights to Rome, search hotels under $100, and optimize "
         "the total trip cost within a $1500 budget", 2),
        ("Plan a trip: search flights, find hotels near the center, "
         "look up restaurants, and optimize within a budget of $2000", 2),
        ("Search for budget flights under $500, find hotels between "
         "$80 and $120, and find restaurants with at least 4 stars", 2),
        ("Find the cheapest flights to Berlin, search for hostels "
         "and hotels, and optimize the total accommodation cost", 2),
        ("Look up flights and hotels to Lisbon, find vegetarian "
         "restaurants, and minimize the total trip spending", 2),

        # SC(3): Entangled, cross-reference, analysis chains
        ("Analyze market trends by searching the web, cross-reference "
         "with social media sentiment, correlate data patterns, and "
         "build a predictive model from the combined insights", 3),
        ("Search for climate data, analyze temperature trends, "
         "cross-reference with economic indicators, and build a "
         "forecast model from the synthesized data", 3),
        ("Search competitor pricing across markets, analyze customer "
         "sentiment, cross-reference trends, and build a strategic "
         "prediction integrating all data sources", 3),
        ("Analyze social media trends, cross-reference with web search "
         "results, correlate sentiment shifts with market data, and "
         "predict future consumer behavior patterns", 3),
        ("Search for financial reports, analyze sentiment from news, "
         "cross-reference with historical market data, and build an "
         "integrated forecast model for investment decisions", 3),
    ]

    data = []
    for query, level in real_queries:
        features = extract_query_features(query, catalog)
        features['_source'] = 'real'
        data.append((features, level))

    return data
