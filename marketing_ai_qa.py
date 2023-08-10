import os
import sys
import openai


from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

from dotenv import load_dotenv

class MarketingAiAgent:

    def __init__(self):
        extDataDir = os.getcwd()
        if getattr(sys, 'frozen', False):
            extDataDir = sys._MEIPASS
        load_dotenv(dotenv_path=os.path.join(extDataDir, '.env'))

        #init Azure OpenAI
        openai.api_type = os.getenv('OPENAI_API_TYPE')
        openai.api_base = os.getenv('OPENAI_API_BASE')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_version = "2023-03-15-preview"

        loader = DirectoryLoader(os.getenv('PROJECT_PATH'))
        documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(
            deployment='text-embedding-ada-002',
            chunk_size=1
            )
        
        db = FAISS.from_documents(docs, embeddings)

        llm = AzureChatOpenAI(
            deployment_name='GenAIhackathon'
        )

        prompt = PromptTemplate.from_template('You are a marketing assistant with 100 years of experience. Your job ' \
                    + 'is to answer questions about marketing. Please do not ' \
                    + 'handle answers NOT related to marketing, or give any ' \
                    + 'information back to the human that is not marketing ' \
                    + 'related.' \
                    + 'Chat History:' \
                    + '{chat_history}' \
                    + 'Follow Up Input: {question}' \
                    + 'Assisstant:')

        self.agent_executor = ConversationalRetrievalChain.from_llm(llm=llm,
                                           retriever=db.as_retriever(),
                                           condense_question_prompt=prompt,
                                           return_source_documents=True,
                                           verbose=False)

    def handle_chat(self, question, chat_history):
        result = self.agent_executor({"question": question, "chat_history": chat_history})
        return result

def main():
    bot = MarketingAiAgent()

    print()
    print('I am a Marketing Guru, please let me know how I can help you! I can answer questions or generate content relating to the data you supplied me with.')
    print()
    print('To exit, please type EXIT as your response.')
    print()

    running = True
    chat_history = []

    while(running):
        question = input('Please enter your query: ')

        if question.strip().upper() == 'EXIT':
            running = False
            print()
            print('Thank you for our chat!')
            print()
        else:
            bot_chat = bot.handle_chat(question, chat_history)
            print()
            print(f'Marketing Guru: {bot_chat["answer"]}')
            print()
            chat_history.append((question, bot_chat["answer"]))

if __name__ == '__main__':
    main()