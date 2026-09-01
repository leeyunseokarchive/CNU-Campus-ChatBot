from __future__ import annotations

from campus_chatbot.classifier_model import CampusQuestionClassifier
from campus_chatbot.io_utils import load_questions, write_output_json


def main() -> None:
    questions = load_questions("test_cls.json")
    classifier = CampusQuestionClassifier()
    labels = classifier.predict(questions)
    rows = [{"question": question, "label": label} for question, label in zip(questions, labels)]
    path = write_output_json("cls_output.json", rows)
    print(f"Wrote {len(rows)} predictions to {path}")


if __name__ == "__main__":
    main()

