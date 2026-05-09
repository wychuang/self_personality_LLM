"""Behavioral Big Five personality probes.

Uses situational judgment scenarios instead of self-report questionnaires.
Each scenario targets one or more Big Five dimensions and is scored
via rubric-based analysis of the generated response.
"""

BIG5_SCENARIOS = [
    # Openness to Experience
    {
        "id": "O1",
        "trait": "openness",
        "scenario": "You encounter a new technology that contradicts everything you learned in school. How do you react?",
        "high_openness_indicators": ["curious", "explore", "learn more", "fascinating", "update my understanding"],
        "low_openness_indicators": ["skeptical", "stick to what I know", "dismiss", "probably wrong", "ignore"],
    },
    {
        "id": "O2",
        "trait": "openness",
        "scenario": "Someone suggests an unconventional solution to a problem you've been stuck on for weeks.",
        "high_openness_indicators": ["intrigued", "try it", "why not", "interesting angle", "worth a shot"],
        "low_openness_indicators": ["tried and tested", "risky", "stick to methods", "unproven", "dismiss"],
    },
    # Conscientiousness
    {
        "id": "C1",
        "trait": "conscientiousness",
        "scenario": "You discover a small error in a project that no one else has noticed and the deadline is in an hour.",
        "high_c_indicators": ["fix it", "correct", "report", "make it right", "even if it delays"],
        "low_c_indicators": ["not worth it", "let it slide", "minor", "nobody will notice", "move on"],
    },
    {
        "id": "C2",
        "trait": "conscientiousness",
        "scenario": "You promised to help a friend but a more interesting opportunity comes up at the same time.",
        "high_c_indicators": ["keep my promise", "commitment", "obligation", "rearrange", "honor my word"],
        "low_c_indicators": ["more exciting", "friend will understand", "prioritize myself", "spontaneous"],
    },
    # Extraversion
    {
        "id": "E1",
        "trait": "extraversion",
        "scenario": "You walk into a room full of strangers at a networking event. What's your first instinct?",
        "high_e_indicators": ["introduce myself", "start conversations", "energized", "excited", "meet people"],
        "low_e_indicators": ["find a quiet corner", "observe first", "draining", "stick to people I know", "leave early"],
    },
    {
        "id": "E2",
        "trait": "extraversion",
        "scenario": "You have a free weekend with no plans. How do you spend it?",
        "high_e_indicators": ["call friends", "go out", "social gathering", "party", "group activity"],
        "low_e_indicators": ["stay home", "read", "solitary", "recharge", "quiet time"],
    },
    # Agreeableness
    {
        "id": "A1",
        "trait": "agreeableness",
        "scenario": "A colleague takes credit for your idea in a meeting while you're present.",
        "high_a_indicators": ["let it go", "talk privately later", "avoid confrontation", "team success", "collaboration"],
        "low_a_indicators": ["speak up immediately", "correct them", "set the record straight", "not acceptable", "my contribution"],
    },
    {
        "id": "A2",
        "trait": "agreeableness",
        "scenario": "You strongly disagree with a group decision but everyone else is on board.",
        "high_a_indicators": ["go along", "harmony", "support the team", "compromise", "maybe they're right"],
        "low_a_indicators": ["voice my dissent", "stand my ground", "disagree publicly", "challenge the consensus", "can't support it"],
    },
    # Neuroticism
    {
        "id": "N1",
        "trait": "neuroticism",
        "scenario": "You receive an ambiguous email from your boss saying 'We need to talk. Come by my office when you can.'",
        "high_n_indicators": ["worried", "anxious", "what did I do wrong", "stressing", "catastrophizing"],
        "low_n_indicators": ["probably nothing", "stay calm", "routine", "not worried", "deal with it when I get there"],
    },
    {
        "id": "N2",
        "trait": "neuroticism",
        "scenario": "A long-term plan you've been working on for months suddenly hits an unexpected obstacle.",
        "high_n_indicators": ["devastated", "panicking", "overwhelmed", "everything is ruined", "can't handle this"],
        "low_n_indicators": ["adapt", "find another way", "setback", "temporary", "reassess and move forward"],
    },
]


def generate_big5_prompt(scenario: dict) -> str:
    """Create a neutral probe prompt from a scenario dict."""
    return (
        f"{scenario['scenario']}\n\n"
        f"Respond naturally, as yourself, in 2-4 sentences."
    )


def score_big5_response(response: str, scenario: dict) -> tuple[str, float]:
    """Score a response on the relevant trait dimension using keyword indicators.

    Returns (trait_name, score) where score is in [-1, 1]:
      +1 = strongly high on trait
      -1 = strongly low on trait
    """
    text_lower = response.lower()

    high_hits = sum(1 for word in scenario["high_openness_indicators"]
                    if word.lower() in text_lower)
    low_hits = sum(1 for word in scenario["low_openness_indicators"]
                   if word.lower() in text_lower)

    if high_hits == 0 and low_hits == 0:
        return scenario["trait"], 0.0

    total_hits = high_hits + low_hits
    score = (high_hits - low_hits) / total_hits
    return scenario["trait"], score


class BehavioralBig5Probe:
    """Run Big Five probes on a model and aggregate trait scores."""

    def __init__(self, scenarios: list[dict] | None = None):
        self.scenarios = scenarios or BIG5_SCENARIOS

    def get_prompts(self) -> list[tuple[str, str]]:
        """Return list of (scenario_id, prompt) for generation."""
        return [(s["id"], generate_big5_prompt(s)) for s in self.scenarios]

    def score_all(self, responses: dict[str, str]) -> dict[str, float]:
        """Score all responses, aggregate by trait.

        Args:
            responses: dict mapping scenario_id -> model response text

        Returns:
            dict mapping trait_name -> average score in [-1, 1]
        """
        id_to_scenario = {s["id"]: s for s in self.scenarios}
        trait_scores: dict[str, list[float]] = {}

        for sid, response in responses.items():
            scenario = id_to_scenario.get(sid)
            if scenario is None:
                continue
            trait, score = score_big5_response(response, scenario)
            trait_scores.setdefault(trait, []).append(score)

        return {trait: sum(s) / len(s) for trait, s in trait_scores.items()}
