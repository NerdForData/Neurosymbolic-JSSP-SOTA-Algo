"""
LLM Client (Groq)
=================
Wraps the Groq API to generate job priority sequences for JSSP instances.

Responsibilities:
  - Send structured prompts to Groq (Llama 3.1 8B)
  - Parse and validate the JSON response
  - Repair any invalid sequences (wrong length, wrong job IDs)
  - Fall back to heuristic sequences if the API fails

This module is the "Neuro" half of the Neuro-Symbolic system.
"""

import os
import json
import random
import logging
from typing import List, Optional
from dotenv import load_dotenv

from src.problem.jssp import JSSPInstance
from src.llm.prompt_builder import build_messages

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence validation & repair
# ---------------------------------------------------------------------------

def _is_valid_sequence(seq: list, n_jobs: int, n_machines: int) -> bool:
    """Check that a sequence has the right length and each job appears n_machines times."""
    if len(seq) != n_jobs * n_machines:
        return False
    if not all(isinstance(x, int) and 0 <= x < n_jobs for x in seq):
        return False
    from collections import Counter
    counts = Counter(seq)
    return all(counts[j] == n_machines for j in range(n_jobs))


def _repair_sequence(seq: list, n_jobs: int, n_machines: int) -> List[int]:
    """
    Repair a malformed sequence by removing excess job IDs and
    padding with missing ones in random positions.
    """
    target = n_machines
    result = []
    counts = {j: 0 for j in range(n_jobs)}

    # Keep valid job IDs up to the target count
    for job_id in seq:
        if 0 <= job_id < n_jobs and counts[job_id] < target:
            result.append(job_id)
            counts[job_id] += 1

    # Add missing job appearances at random positions
    for j in range(n_jobs):
        needed = target - counts[j]
        for _ in range(needed):
            pos = random.randint(0, len(result))
            result.insert(pos, j)

    return result[:n_jobs * n_machines]


# ---------------------------------------------------------------------------
# Heuristic fallbacks (used when LLM call fails)
# ---------------------------------------------------------------------------

def _spt_sequence(instance: JSSPInstance) -> List[int]:
    """Shortest Processing Time: jobs with less total work go first."""
    jobs_sorted = sorted(instance.jobs, key=lambda j: j.total_processing_time)
    seq = []
    for _ in range(instance.num_machines):
        seq.extend(j.job_id for j in jobs_sorted)
    random.shuffle(seq)  # add slight randomness
    return seq


def _lpt_sequence(instance: JSSPInstance) -> List[int]:
    """Longest Processing Time: keeps machines busy."""
    jobs_sorted = sorted(instance.jobs,
                         key=lambda j: j.total_processing_time, reverse=True)
    seq = []
    for _ in range(instance.num_machines):
        seq.extend(j.job_id for j in jobs_sorted)
    return seq


def _edd_sequence(instance: JSSPInstance) -> List[int]:
    """Earliest Due Date: minimizes tardiness."""
    jobs_sorted = sorted(instance.jobs,
                         key=lambda j: (j.due_date or float("inf")))
    seq = []
    for _ in range(instance.num_machines):
        seq.extend(j.job_id for j in jobs_sorted)
    return seq


def _random_sequence(instance: JSSPInstance) -> List[int]:
    """Random sequence for diversity."""
    seq = list(range(instance.num_jobs)) * instance.num_machines
    random.shuffle(seq)
    return seq


def generate_heuristic_sequences(instance: JSSPInstance,
                                  n: int = 5) -> List[List[int]]:
    """
    Generate n heuristic sequences without using the LLM.
    Used as fallback and to fill remaining population slots.
    """
    generators = [_spt_sequence, _lpt_sequence, _edd_sequence, _random_sequence]
    sequences = []
    for i in range(n):
        gen = generators[i % len(generators)]
        sequences.append(gen(instance))
    return sequences


# ---------------------------------------------------------------------------
# Groq LLM client
# ---------------------------------------------------------------------------

class GroqLLMClient:
    """
    Calls Groq API (Llama 3.1 8B) to generate job priority sequences.

    Usage:
        client = GroqLLMClient()
        sequences = client.generate_sequences(instance, n=5)
    """

    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model   = model or self.DEFAULT_MODEL

        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY not found. LLM calls will use heuristic fallback. "
                "Set GROQ_API_KEY in your .env file to enable LLM seeding."
            )
            self._client = None
        else:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                logger.info(f"Groq client initialised (model={self.model})")
            except ImportError:
                raise ImportError("Run: pip install groq")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def generate_sequences(
        self,
        instance: JSSPInstance,
        n: int = 5,
        temperature: float = 0.7,
    ) -> List[List[int]]:
        """
        Generate n priority sequences for the given instance.

        If the Groq API is unavailable or returns invalid data,
        falls back to heuristic sequences automatically.

        Args:
            instance:    JSSP instance to generate sequences for
            n:           Number of sequences to generate
            temperature: LLM temperature (higher = more diverse)

        Returns:
            List of n valid priority sequences
        """
        if not self.is_available:
            logger.info("Using heuristic fallback (no API key)")
            return generate_heuristic_sequences(instance, n)

        try:
            return self._call_api(instance, n, temperature)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}. Falling back to heuristics.")
            return generate_heuristic_sequences(instance, n)

    def _call_api(
        self,
        instance: JSSPInstance,
        n: int,
        temperature: float,
    ) -> List[List[int]]:
        """Make the actual API call and parse the response."""
        messages = build_messages(instance, n_sequences=n)

        logger.info(f"Calling Groq API (model={self.model}, n={n})...")
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        logger.info(f"LLM response received ({len(raw)} chars)")

        return self._parse_response(raw, instance, n)

    def _parse_response(
        self,
        raw: str,
        instance: JSSPInstance,
        n: int,
    ) -> List[List[int]]:
        """
        Parse the JSON response from the LLM.
        Validates and repairs each sequence.
        Falls back to heuristics for any invalid sequences.
        """
        n_jobs     = instance.num_jobs
        n_machines = instance.num_machines

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:300]}")

        raw_sequences = data.get("sequences", [])
        reasoning     = data.get("reasoning", "")
        if reasoning:
            logger.info(f"LLM reasoning: {reasoning[:200]}")

        valid_sequences = []
        for i, seq in enumerate(raw_sequences):
            seq = [int(x) for x in seq]  # ensure ints

            if _is_valid_sequence(seq, n_jobs, n_machines):
                valid_sequences.append(seq)
                logger.debug(f"  Sequence {i}: valid")
            else:
                repaired = _repair_sequence(seq, n_jobs, n_machines)
                if _is_valid_sequence(repaired, n_jobs, n_machines):
                    valid_sequences.append(repaired)
                    logger.debug(f"  Sequence {i}: repaired")
                else:
                    logger.debug(f"  Sequence {i}: discarded (could not repair)")

        # Fill remaining slots with heuristics if LLM didn't give enough
        if len(valid_sequences) < n:
            needed = n - len(valid_sequences)
            logger.info(f"  Filling {needed} slots with heuristic sequences")
            valid_sequences.extend(generate_heuristic_sequences(instance, needed))

        return valid_sequences[:n]