import os

from upload_data import db
from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

DOC_DIR = os.path.dirname("../doc/")
INTENT_PROMPT_TEMPLATE = os.path.join(DOC_DIR, "parse_intent.txt")
INTENT_LIST_TXT = os.path.join(DOC_DIR, "intent_list.txt")
CHATBOT_PROMPT_TEMPLATE = os.path.join(DOC_DIR, "chatbot_prompt_template.txt")


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )


llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")


parse_intent_chain = create_chain(
    llm=llm,
    template_path=INTENT_PROMPT_TEMPLATE,
    output_key="text",
)
kakao_sync_step1_chain = create_chain(
    llm=llm,
    template_path=CHATBOT_PROMPT_TEMPLATE,
    output_key="text",
)
kakao_social_step1_chain = create_chain(
    llm=llm,
    template_path=CHATBOT_PROMPT_TEMPLATE,
    output_key="text",
)
kakaotalk_channel_step1_chain = create_chain(
    llm=llm,
    template_path=CHATBOT_PROMPT_TEMPLATE,
    output_key="text",
)
default_chain = ConversationChain(llm=llm, output_key="text")


def generate_answer(user_message) -> dict[str, str]:
    context = dict(user_message=user_message)
    context["query"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)

    # intent = parse_intent_chain(context)["intent"]
    intent = parse_intent_chain.run(context)

    if intent == "KAKAO_SYNC":
        result = db.similarity_search(query=user_message, where={"service": "KAKAO_SYNC"}, limit=5)
        print(result)
        context["base_data"] = db.similarity_search(query=user_message, where={"service": "KAKAO_SYNC"}, limit=5)
        answer = kakao_sync_step1_chain.run(context)
    elif intent == "KAKAO_SOCIAL":
        context["base_data"] = db.similarity_search(query=user_message, where={"service": "KAKAO_SOCIAL"}, limit=5)
        answer = kakao_social_step1_chain.run(context)
    elif intent == "KAKAOTALK_CHANNEL":
        context["base_data"] = db.similarity_search(query=user_message, where={"service": "KAKAOTALK_CHANNEL"}, limit=5)
        answer = kakaotalk_channel_step1_chain.run(context)
    else:
        answer = default_chain.run(context)

    return {"answer": answer}


generate_answer("카카오 싱크 시작하는 법 알려줘")