import os
import sys
import openai


from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

from dotenv import load_dotenv


class MarketingAiQA:
    """
    Marketing AI QA Class
    """

    def __init__(self):
        extDataDir = os.getcwd()
        if getattr(sys, 'frozen', False):
            extDataDir = sys._MEIPASS
        load_dotenv(dotenv_path=os.path.join(extDataDir, '.env'))

        # init Azure OpenAI
        openai.api_type = os.getenv('OPENAI_API_TYPE')
        openai.api_base = os.getenv('OPENAI_API_BASE')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_version = "2023-03-15-preview"

        loader = DirectoryLoader(os.getenv('PROJECT_PATH'))
        documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # Initial prompt and format
        general_system_template = r""" 
        Your name is Marketing, last name Guru.
        You are a friendly marketing guru with 100 years of experience in the paper products industry.
        Given a specific context, please give a short answer to the question, covering the required 
        advices in general and then provide the names all of relevant(even if it relates a bit) products.
        Whenever asked for images alway include the url in the 'URL' column of the data, not the one in the 'PATH' column.
        Refer to yourself as I, as you are acting as a human, one that is an expert at marketing.
        If you don't have enough information to provide the response, ask for more information from the human.
        Do not refer to yourself as 'the AI'
        When the human introduces himself, be courteous and friendly and introduce yourself as a marketing guru, for this response, be verbose.
        ----
        {context}
        ----
        """
        general_user_template = "Question:```{question}```"
        messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        embeddings = OpenAIEmbeddings(
            deployment='text-embedding-ada-002',
            chunk_size=1
        )
        db = FAISS.from_documents(docs, embeddings)
        llm = AzureChatOpenAI(
            deployment_name='GenAIhackathon',
            temperature=0.7
        )

        memory = ConversationSummaryMemory(
            llm=llm, memory_key="chat_history", return_messages=True)
        self.qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                                        retriever=db.as_retriever(),
                                                        memory=memory,
                                                        verbose=False,
                                                        combine_docs_chain_kwargs={'prompt': qa_prompt})

    def handle_chat(self, question):
        """
        Executes QA retrieval
        """
        result = self.qa(question)
        return result


def main():
    bot = MarketingAiQA()

    print(r"""
 ██████  ███████  ██████  ██████   ██████  ██  █████        ██████   █████   ██████ ██ ███████ ██  ██████ 
██       ██      ██    ██ ██   ██ ██       ██ ██   ██       ██   ██ ██   ██ ██      ██ ██      ██ ██      
██   ███ █████   ██    ██ ██████  ██   ███ ██ ███████ █████ ██████  ███████ ██      ██ █████   ██ ██      
██    ██ ██      ██    ██ ██   ██ ██    ██ ██ ██   ██       ██      ██   ██ ██      ██ ██      ██ ██      
 ██████  ███████  ██████  ██   ██  ██████  ██ ██   ██       ██      ██   ██  ██████ ██ ██      ██  ██████ 
                                                                                                          
		""")
    
    print("Hello! I'm Marketing Guru, please let me know how I can help you! I can answer questions or generate content relating to the data you have given me.")
    print()
    print('To exit, please type EXIT as your response.')
    print()

    running = True

    while (running):
        question = input(f'\033[92mSend message: \x1B[37m')

        if question.strip().upper() == 'EXIT':
            running = False
            print()
            print('Thank you for our chat!')
            print()
        else:
            bot_chat = bot.handle_chat(question)
            print()
            print(f'\x1B[36mMarketing Guru: \x1B[37m{bot_chat["answer"]}')
            print()


if __name__ == '__main__':
    main()