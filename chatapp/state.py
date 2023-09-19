# state.py
import os
import reflex as rx
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = ""

def read_prompt_template() -> str:
    file_path = "./doc/chatbot_prompt_template.txt"
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def load_base_data() -> str:
    file_path = "./doc/project_data_카카오싱크.txt"
    f = open(file_path, 'r')
    lines = f.readlines()
    return ''.join(lines)


def generate_answer(query: str):
    chatbot_llm = ChatOpenAI(temperature=0.3, max_tokens=8192, model='gpt-3.5-turbo-16k')
    chatbot_base_data = load_base_data()
    chatbot_prompt_template = ChatPromptTemplate.from_template(template=read_prompt_template())
    chatbot_chain = LLMChain(llm=chatbot_llm, prompt=chatbot_prompt_template, output_key="answer")

    result = chatbot_chain(dict(
        query=query,
        base_data=chatbot_base_data
    ))

    return result['answer']

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

        answer = generate_answer(self.question)
        self.question = ""


        self.chat_history[-1] = (
            self.chat_history[-1][0],
            answer
        )
