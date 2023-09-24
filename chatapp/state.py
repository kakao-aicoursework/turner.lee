# state.py
from chatapp.llm import generate_answer
import reflex as rx


class State(rx.State):
    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]]

    def answer(self):
        # Add to the answer as the chatbot responds.
        answer = ""
        self.chat_history.append((self.question, answer))

        # Yield here to clear the frontend input before continuing.
        yield

        answer = generate_answer(self.question)["answer"]
        self.question = ""


        self.chat_history[-1] = (
            self.chat_history[-1][0],
            answer
        )
