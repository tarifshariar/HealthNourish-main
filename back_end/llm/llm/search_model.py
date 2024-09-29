import uuid
import regex as re

from back_end.llm.chat_model import LLM
from back_end.llm.PhiPrompt import PhiPrompt

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader as PDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import (
    RunnableLambda
)

from back_end.web_search.SearchClient import MultiSearch

class Chain:
    def __init__(self, model_id: str, api_key: dict, embedding_model_id: str = 'all-MiniLM-L6-v2', device: str = 'cuda'):
        self.model_id = model_id
        self.phi_prompt_builder = PhiPrompt()
        self.llm= LLM(model_id=model_id, device=device).get_llm()
        self.search = MultiSearch(api_key)

        # self.embedding = HuggingFaceEmbeddings(model_name=embedding_model_id, model_kwargs={'device': 'cuda'})
        # self.vector_store = InMemoryVectorStore(self.embedding)

    # def load_pdf(self, path):
    #     loader = PDFLoader(path)
    #     pages = loader.load_and_split()
    #     self.vector_store.add_documents(pages)
    #     print('Page Added')

    def _prompt_template(self, system_prompt_text: str, user_prompt_text: str, input_vars: []) -> PromptTemplate:
        system_prompt = self.phi_prompt_builder.system_prompt(system_prompt_text)
        user_prompt = self.phi_prompt_builder.user_prompt(user_prompt_text)
        assistant_prompt = self.phi_prompt_builder.assistant_prompt()

        prompt = system_prompt + ' ' + user_prompt + ' ' + assistant_prompt

        return PromptTemplate(template=prompt, input_variables=input_vars)

    def _search_prompt(self):
        system_prompt_text = """You are a helpful assistant. You will assist me by generating queries and returning \
the queries in a list format to search the web about recipes for given context."""

        user_prompt_text = """
        Generate 3 queries to find food recipe for context: '{prompt}'
        """

        return self._prompt_template(system_prompt_text, user_prompt_text, ["prompt"])

    def _web_search(self, queries: []):

        search_results = [self.search.search(q) for q in queries]
        return search_results

    def search_chain(self, prompt: str) -> list:
        search_prompt = self._search_prompt()
        chain = ((search_prompt
                  | self.llm
                  | StrOutputParser()
                  | (lambda x: (x.split("<|assistant|>")[-1]).strip())
                  | (lambda x: (re.findall(r'[1-9]\.\s?"(.*)"\n?', x)))
                  )
                 | RunnableLambda(self._web_search)
                 )
        
        res = chain.invoke({"prompt": prompt})

        return res