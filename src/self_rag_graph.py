import os
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List
from typing_extensions import TypedDict


def get_rag_answer(question_content: str):

    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
        """

        question: str
        generation: str
        documents: List[str]

    ### Nodes

    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    ### Edges

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            # TODO no relevant doc should call llm rather than re generate.
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    os.environ["OPENAI_API_KEY"] = (
        "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"
    )

    # shell: export USER_AGENT="YourAppName/1.0"
    os.environ["USER_AGENT"] = "YourAppName/1.0"

    ### 1. Retriever
    print("1. Retriever")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    ### 2. Retrieval Grader
    print("2. Retrieval Grader")

    from langchain_core.prompts import ChatPromptTemplate

    # fix warning : from langchain_core.pydantic_v1 import BaseModel, Field
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI

    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    question = "agent memory"
    # fix warning: docs = retriever.get_relevant_documents(question)
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

    ### 3. Generate
    print("3. Generate")

    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": docs, "question": question})
    print(generation)

    ### 4. Hallucination Grader
    print("4. Hallucination Grader")

    # Data model
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )

    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
            ),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    hallucination_grader.invoke({"documents": docs, "generation": generation})

    ### 5. Answer Grader
    print("5. Answer Grader")

    # Data model
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )

    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "User question: \n\n {question} \n\n LLM generation: {generation}",
            ),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    answer_grader.invoke({"question": question, "generation": generation})

    ### 6. Question Re-writer
    print("6. Question Re-writer")

    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    question_rewriter.invoke({"question": question})

    ### 7. build graph
    print("7. build Graph")

    from langgraph.graph import END, StateGraph, START

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile
    app = workflow.compile()

    print(app.get_graph().draw_mermaid())

    from IPython.display import Image, display
    from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

    graph_img = Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )

    with open("graph_img.png", "wb") as f:
        f.write(graph_img.data)
    print("Image saved as graph_img.png")

    sys.exit(0)

    from pprint import pprint

    # Run
    inputs = {"question": question_content}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"])
    return value["generation"]

if __name__ == "__main__":

    get_rag_answer("Explain how the different types of agent memory work?")
