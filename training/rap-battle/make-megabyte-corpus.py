"""Expand fresh CC0 character verses into reproducible training corpora."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
replies = []
challenges = {}
for row in (root / "megabyte-verses.txt").read_text().splitlines():
    role, topic, first, second, reply, punchline, endings = row.split("|")
    replies.extend([f"[{role}]", f"opening | {first} | {second}"])
    challenges[role] = {"train": [first + "\n" + second], "validation": []}
    for pair in endings.split(";"):
        word, rhyme = pair.split()
        for setup, payoff in [
            (f"Bring your next verse to the edge of the {word}",
             f"I'll finish the hook before you reach the {rhyme}."),
            (f"Try keeping time when I bring out the {word}",
             f"I'll leave you a bar while I work in the {rhyme}."),
        ]:
            challenges[role]["train"].append(setup + "\n" + payoff)
            replies.append(f"{word} | {setup} | {payoff} | Your {{cue}} got a turn by the edge of the {word} | The crowd came back when I brought out the {rhyme}.")
    replies.append(f"{topic} | {first} | {second} | {reply} | {punchline}")
    challenges[role]["validation"].append(f"Let's hear your next line about the {topic}\nI'll give you a turn and then answer in kind.")
(root / "megabyte-replies.txt").write_text("\n".join(replies) + "\n")
(root / "megabyte-challenges.json").write_text(json.dumps(challenges, indent=2) + "\n")
