from core.utils import check_exit_intent, contains_phrase, format_conversation_history, compress_practice_summary
from langchain_core.messages import HumanMessage, AIMessage

def test_contains_phrase():
    assert contains_phrase("Time and Work", "time and work")
    assert not contains_phrase("Corporate Finance", "rate")
    assert contains_phrase("This is a rate problem", "rate")

def test_check_exit_intent():
    # False positives from audit
    assert not check_exit_intent("I've done 3 practice sets, give me more")
    assert not check_exit_intent("no thanks, keep going")

    # True positives
    assert check_exit_intent("bye")
    assert check_exit_intent("thank you")

    # Length constraint
    assert not check_exit_intent("thanks for all the help you have given me today")

    # A continuation word ("more"/"another") earlier in the message must not
    # suppress an explicit, unambiguous exit word at the end.
    assert check_exit_intent("no more questions, bye")
    assert check_exit_intent("not another one, quit")

def test_format_conversation_history():
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi"),
        HumanMessage(content="How are you?"),
        AIMessage(content="I am fine"),
        HumanMessage(content="Current query")
    ]
    formatted = format_conversation_history(messages, max_turns=1)
    assert "User: Hello" not in formatted
    assert "User: How are you?" in formatted
    assert "Assistant: I am fine" in formatted
    assert "User: Current query" not in formatted

def test_compress_practice_summary():
    messages = [HumanMessage(content=str(i)) for i in range(5)]
    assert compress_practice_summary(messages) == ""

    messages = [HumanMessage(content=str(i)) for i in range(10)]
    assert compress_practice_summary(messages) != ""
