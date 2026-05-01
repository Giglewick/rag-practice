from langchain_core.prompts import ChatPromptTemplate

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
    "answer": answer.content,
    "context": context
})

print("\nLLM evaluation:")
print(eval_result.content)