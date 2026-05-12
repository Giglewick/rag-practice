from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

answer = """
Q-learning is a reinforcement learning method.
"""

context = """
Q-learning is a reinforcement learning algorithm introduced by Watkins.
"""

evaluation_prompt = ChatPromptTemplate.from_template("""
Answer:
{answer}

Context:
{context}

Does the answer only use information from the context?
Answer with: YES or NO and explain.
""")

eval_chain = evaluation_prompt | llm

eval_result = eval_chain.invoke({
    "answer": answer,
    "context": context
})

print("\nLLM evaluation:")
print(eval_result.content)