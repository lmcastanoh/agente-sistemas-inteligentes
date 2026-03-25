from __future__ import annotations

from langchain_openai import ChatOpenAI

from prompts import CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_USER_TEMPLATE
from schemas import IntentClassification


def route_from_intent(intent: IntentClassification) -> str:
    return "RAG(retrieve→agent_reason→generate→evaluate)" if intent.needs_retrieval else "GENERAL(answer_general)"


def main():
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0).with_structured_output(IntentClassification)

    queries = [
        "What is the horsepower of Toyota Hilux 2024 GR Sport?",
        "Show the torque and transmission for Mazda CX-5 2023 Signature.",
        "How many airbags does Nissan Frontier PRO-4X 2022 have?",
        "Summarize the technical datasheet for Hyundai Tucson 2025.",
        "Give me a short summary of safety and drivetrain for Kia Sportage 2024.",
        "Compare Hilux 2024 vs Ranger 2024 in engine, torque, fuel economy, and dimensions.",
        "Compare Corolla Cross Hybrid 2025 and CX-30 2025 trims.",
        "What is torque in a vehicle?",
        "Explain differences between ABS and ESC.",
        "What does CVT mean and when is it useful?",
        "Tell me specs for Corolla Cross.",
        "Which one is better between Hilux and Ranger?",
    ]

    print("=== Intent Routing Test (12 queries) ===")
    for i, q in enumerate(queries, start=1):
        intent: IntentClassification = llm.invoke(
            [
                ("system", CLASSIFIER_SYSTEM_PROMPT),
                ("user", CLASSIFIER_USER_TEMPLATE.format(question=q)),
            ]
        )
        print(f"{i:02d}. {q}")
        print(
            f"    intent={intent.intent} | needs_retrieval={intent.needs_retrieval} | route={route_from_intent(intent)}"
        )
        if intent.clarification_question:
            print(f"    clarification_question={intent.clarification_question}")


if __name__ == "__main__":
    main()
