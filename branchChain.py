from dotenv import load_dotenv
from langchain.schema.runnable import RunnableBranch
from langchain.prompt import ChatPromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()
model=ChatOpenAI(model="gpt-4o")

positive_feedback=ChatPromptTemplate.from_messages(
    [
        ("system","you are a smart assistant"),
        ("human", "give a positive reply to this feedback {feedback}"),
    ]
)
negative_feedback=ChatPromptTemplate.from_messages(
    [
        ("system","you are a smart assistant"),
        ("human", "give a negative reply to this feedback {feedback}"),
    ]
)
neutral_feedback=ChatPromptTemplate.from_messages(
    [
        ("system","you are a smart assistant"),
        ("human", "give a neutral reply to this feedback {feedback}"),
    ]
)
escalted_feedback=ChatPromptTemplate.from_messages(
    [
        ("system","you are a smart assistant"),
        ("human", "give a escalated reply to this feedback {feedback}"),
    ]
)
classify_feedback=ChatPromptTemplate.from_messages(
    [
        ("system", "you are a smart assistant"),
        ("human", "classify the given feedback has positive, negative, neytral, escalted"),
    ]
)

branches=RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback|model|StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback|model|StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_feedback|model|StrOutputParser()
    ),
    (
        lambda x: "escalted" in x,
        escalted_feedback|model|StrOutputParser()
    )
)
classify_chain=classify_feedback|model|StrOutputParser()
chain=classify_chain|branches
reviews="this product is terrible, it broke after first use."
result=chain.invoke({"feedback": reviews})
print(result)
#some feedbacks 