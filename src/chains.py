def build_rag_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def rag_chain(query: str, chat_history=None):
        docs = retriever.invoke(query)
        
        if not docs:
            return "Not found in document."
        
        context = "\n".join([d.page_content for d in docs])
        
        history = ""
        if chat_history and len(chat_history) > 0:
            history = "Previous conversation:\n"
            for msg in chat_history[-4:]:
                history += f"{msg['role']}: {msg['content']}\n"
            history += "\n"
        
        prompt = f"""Answer the question. If the answer is in the document, use it. 
If not in the document but you can answer from general knowledge, answer but say: "(Note: This is not from the document)"

{history}Document Context: {context}

Question: {query}"""
        
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    return rag_chain