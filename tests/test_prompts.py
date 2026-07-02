"""Regression tests: PromptTemplate validates that every {placeholder} in the
template string has a matching entry in input_variables (and vice versa), so
these catch a typo'd or removed variable before it ever reaches an LLM call."""
from langchain_core.prompts import PromptTemplate

from agents.prompts import (
    FEEDBACK_PROMPT,
    PRACTICE_QUESTIONS_PROMPT,
    ROUTER_PROMPT,
    STUDY_PLAN_PROMPT,
)


def test_router_prompt_template_is_valid():
    PromptTemplate(template=ROUTER_PROMPT, input_variables=["question"])


def test_study_plan_prompt_template_is_valid():
    PromptTemplate(
        template=STUDY_PLAN_PROMPT, input_variables=["question", "context", "history"]
    )


def test_practice_questions_prompt_template_is_valid():
    PromptTemplate(
        template=PRACTICE_QUESTIONS_PROMPT,
        input_variables=["question", "context", "history", "current_questions", "previous_summary"],
    )


def test_feedback_prompt_template_is_valid():
    PromptTemplate(
        template=FEEDBACK_PROMPT,
        input_variables=["question", "context", "history", "mock_analysis", "weak_areas"],
    )
